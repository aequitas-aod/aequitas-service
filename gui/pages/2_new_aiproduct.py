# streamlit: page_name = "New AI Product"
import os
import sys
import yaml
import json
import asyncio
import pandas as pd
import streamlit as st
from streamlit_ace import st_ace
from collections import defaultdict
from utils import (
    populate_stages,
    get_application_domains,
    get_ai_tasks,
    get_ai_type_of_use,
    label_for_iri,
    get_fairness_concerns,
    get_fairness_notions,
    get_fairness_metrics,
    get_mitigation_techniques_for_concern,
    render_cascade_checkbox,
    render_cascade_question,
    render_competency_questions,
    load_method_content,
    load_aipc_config,
    import_from_path,
    get_function_source
)
from dotenv import load_dotenv, find_dotenv

from langchain_mcp_adapters.client import MultiServerMCPClient
from langchain_mcp_adapters.tools import load_mcp_tools
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, MessagesState, START, END
from langgraph.prebuilt import ToolNode

_ = load_dotenv(find_dotenv())

st.set_page_config(layout="wide", page_title="New AI Product")  # , page_icon="📊"
st.title("Compliance Assessment tool")
# st.sidebar.header("DataFrame Demo")

# Enlarges the tab labels (st.tabs renders them small by default).
st.markdown(
    """
    <style>
    .stTabs [data-baseweb="tab-list"] button [data-testid="stMarkdownContainer"] p {
        font-size: 20px;
        font-weight: 600;
    }
    .stButton > button {
        border-radius: 20px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


def render_section_header(text):
    """Section title styled larger than the default st.write() body text."""
    st.markdown(
        f"<p style='font-size:22px; font-weight:600;'>{text}</p>",
        unsafe_allow_html=True,
    )

current_folder = os.path.dirname(os.path.abspath(__file__))
parent_folder = os.path.dirname(current_folder)
parent_folder = os.path.dirname(parent_folder)
pipeline_definitions_folder = os.path.join(
    parent_folder, "framework/temlops/config/pipeline_definitions.yaml"
)
USE_CASES_FOLDER = os.path.join(parent_folder, "framework/temlops/use_cases")
TOOLS_CATALOG_FOLDER = os.path.join(parent_folder, "tools_catalog")
sys.path.append(USE_CASES_FOLDER)


# Step 1: definition of the necessary operations (active/inactive ops)
with open(pipeline_definitions_folder, "r") as yaml_file:
    pipeline_configs = yaml.safe_load(yaml_file)
    requirements_dimensions = list(
        map(lambda x: x.capitalize(), pipeline_configs["requirements_dimensions"])
    )


def fairness_settings():
    name_col, desc_col = st.columns(2)
    with name_col:
        ai_prod_name = st.text_area(
            "AI product name", value="Bias free AI assisted recruiting system"
        )
    with desc_col:
        ai_prod_desc = st.text_area(
            "AI product description",
            value="""The software supports the recruitment process by recommending the best candidates for a given job position and, conversely, suggesting the most suitable positions for candidates.""",
        )
    domain_col, task_col, use_type_col = st.columns(3)
    with domain_col:
        application_domain = st.selectbox(
            "Application Domain",
            get_application_domains(),
            format_func=label_for_iri,
        )
    with task_col:
        ai_task = st.selectbox(
            "AI Task",
            get_ai_tasks(),
            format_func=label_for_iri,
        )
    with use_type_col:
        ai_type_of_use = st.selectbox(
            "AI Type of Use",
            get_ai_type_of_use(),
            format_func=label_for_iri,
        )
    st.session_state["application_domain"] = application_domain
    st.session_state["ai_task"] = ai_task
    st.session_state["ai_type_of_use"] = ai_type_of_use

    render_section_header("Questions")
    render_competency_questions()

    return ai_prod_name, ai_prod_desc


def lifecycle_stages():
    render_section_header("AI system's stages and operations")
    recommended_group = st.session_state.get("recommended_mitigation_group")
    populate_stages(
        pipeline_configs, createview=True, recommended_group=recommended_group
    )


# Step 1b: cascading fairness requirements for the selected AI Type of Use.
# Checking a Fairness Concern reveals its Fairness Notions; each Notion's
# expander shows its Fairness Metrics and the concern's Mitigation
# Techniques (via MITIGATION_TECHNIQUE_FOR_CONCERN_QUERY) side by side.
def fairness_requirements_section():
    st.write("Relevant fairness requirements for the selected AI Type of Use")
    ai_type_of_use_iri = st.session_state.get("ai_type_of_use")
    if not ai_type_of_use_iri:
        st.info("Select an AI Type of Use above to see relevant fairness concerns.")
        return

    concerns = get_fairness_concerns(ai_type_of_use_iri)
    if not concerns:
        st.info("No fairness concerns found in the ontology for this AI Type of Use.")
        return

    selected_metrics = []
    selected_mitigation_techniques = []
    concerns_notions = defaultdict(list)
    concerns_notions_vis = defaultdict(list)
    for concern in concerns:
        for notion in get_fairness_notions(concern["iri"]):
            concerns_notions[notion["label"]].append(concern["label"])  # notion

    for concern in concerns:
        # with st.expander(
        #    f"{concern['label']}. {concern['definition']}", expanded=False
        # ):

        for notion in get_fairness_notions(concern["iri"]):
            notion_metrics = get_fairness_metrics(notion["iri"])
            if notion["label"] not in concerns_notions_vis and len(notion_metrics) > 0:
                concerns_notions_vis[notion["label"]].append(
                    str(concern["iri"]).split("#")[-1]
                )  # notion
                with st.expander(f"{notion['label']}", expanded=False):
                    concern_badges = "".join(
                        f"""<span style="
                            display:inline-block;
                            background-color:#e0e7ff;
                            color:#3730a3;
                            border-radius:12px;
                            padding:2px 10px;
                            margin:2px 6px 2px 0;
                            font-size:14px;
                            font-weight:600;
                        ">{concern_label}</span>"""
                        for concern_label in concerns_notions[notion["label"]]
                    )
                    st.markdown(
                        f"""
                        <div style="margin-bottom:10px; display:flex; align-items:center; flex-wrap:wrap; gap:4px;">
                            <span style="font-size:12px; color:#6b7280; margin-right:4px;">Arises from:</span>
                            {concern_badges}
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )

                    metrics_col, mitigation_col = st.columns(2)
                    with metrics_col:
                        st.markdown(
                            "<span style='font-size:12px; color:#6b7280;'>Metrics</span>",
                            unsafe_allow_html=True,
                        )
                        for metric in notion_metrics:
                            metric_key = f"metric_{concern['iri']}_{notion['iri']}_{metric['iri']}"
                            metric_checked = render_cascade_checkbox(
                                metric["label"],
                                key=metric_key,
                                level=0,
                            )
                            if metric_checked:
                                selected_metrics.append(metric["iri"])

                    with mitigation_col:
                        st.markdown(
                            "<span style='font-size:12px; color:#6b7280;'>Mitigation techniques</span>",
                            unsafe_allow_html=True,
                        )
                        for technique in get_mitigation_techniques_for_concern(
                            concern["iri"]
                        ):
                            technique_key = f"technique_{concern['iri']}_{notion['iri']}_{technique['iri']}"
                            technique_checked = render_cascade_checkbox(
                                technique["label"],
                                key=technique_key,
                                level=0,
                            )
                            if technique_checked:
                                selected_mitigation_techniques.append(technique["iri"])

                    notion_key = str(notion["iri"]).split("#")[-1]
                    show_custom_metric_key = f"show_custom_metric_{notion_key}"
                    if show_custom_metric_key not in st.session_state:
                        st.session_state[show_custom_metric_key] = False

                    btn_col, _ = st.columns([3, 9])
                    with btn_col:
                        if st.button(
                            "➕ Add custom metric",
                            key=f"add_custom_metric_btn_{notion_key}",
                        ):
                            st.session_state[show_custom_metric_key] = (
                                not st.session_state[show_custom_metric_key]
                            )

                    if st.session_state[show_custom_metric_key]:
                        with st.expander("Define custom metric", expanded=True):
                            st.text_input(
                                "Custom metric name",
                                key=f"custom_metric_name_{notion_key}",
                            )
                            st.text_area(
                                "Custom metric code",
                                key=f"custom_metric_code_{notion_key}",
                                height=220,
                                placeholder=(
                                    "def compute_metric(y_true, y_pred, sensitive_features):\n"
                                    "    # implement the custom fairness metric\n"
                                    "    return value"
                                ),
                            )

    st.session_state["selected_fairness_metrics"] = selected_metrics
    st.session_state["selected_mitigation_techniques"] = selected_mitigation_techniques


# Group -> column color, in pre/in/post-processing order, per
# map_mitigation_recommendations()'s "group" field.
MITIGATION_GROUP_STYLES = {
    "Pre-processing": {"bg": "#fef9c3", "border": "#eab308", "text": "#854d0e"},
    "In-processing": {"bg": "#dcfce7", "border": "#10b981", "text": "#065f46"},
    "Post-processing": {"bg": "#dbeafe", "border": "#3b82f6", "text": "#1e3a8a"},
}


# Renders the recommended mitigation `category` (e.g. "A") as a 3-column
# row -- one column per group in MITIGATION_GROUP_STYLES -- highlighting
# only the column matching the category's group (looked up via
# map_mitigation_recommendations()) and leaving the other two muted.
def render_recommended_mitigation_category(category):
    category_info = {m["category"]: m for m in map_mitigation_recommendations()}.get(
        category
    )
    group = category_info["group"] if category_info else None
    description = category_info["description"] if category_info else ""

    # Read by lifecycle_stages() to grey out / disable the operations
    # belonging to stages that precede this recommended group.
    st.session_state["recommended_mitigation_category"] = category
    st.session_state["recommended_mitigation_group"] = group

    columns = st.columns(len(MITIGATION_GROUP_STYLES))
    for col, (group_name, style) in zip(columns, MITIGATION_GROUP_STYLES.items()):
        with col:
            if group_name == group:
                st.markdown(
                    f"""
                    <div style="
                        margin:8px 0 16px;
                        padding:10px 16px;
                        background-color:{style['bg']};
                        border-left:4px solid {style['border']};
                        border-radius:6px;
                        min-height:78px;
                    ">
                        <div style="font-size:12px; color:{style['text']}; font-weight:700; text-transform:uppercase; margin-bottom:6px;">
                            {group_name}
                        </div>
                        <div style="font-size:16px; color:{style['text']};">
                            ➜ <strong style="font-size:18px;">{category}</strong>  {description}
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
            else:
                st.markdown(
                    f"""
                    <div style="
                        margin:8px 0 16px;
                        padding:10px 16px;
                        background-color:#f9fafb;
                        border-left:4px solid #e5e7eb;
                        border-radius:6px;
                        min-height:88px;
                    ">
                        <div style="font-size:12px; color:#9ca3af; font-weight:700; text-transform:uppercase;">
                            {group_name}
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )


# Renders `questions` as a top-to-bottom decision tree: each question is a
# render_cascade_question() row (question card + horizontal radio buttons).
# A question only appears once every dependency it declares in "depends_on"
# is satisfied by a prior answer ("depends_on" may be a single {q, value}
# dict or a list of them, meaning "any of these"). Indentation depth follows
# the actual answered path that unlocked the question, not just its
# structural position in the list. Runs a small fixed-point loop over
# `questions` so newly-unlocked questions get rendered in the same pass they
# become eligible, however many hops deep they are.
def render_resource_aware_flow(questions):
    by_id = {str(q["id"]): q for q in questions}
    answers = {}
    rendered = set()

    def dependency_list(question):
        deps = question.get("depends_on") or []
        deps = deps if isinstance(deps, list) else [deps]
        return [d for d in deps if d.get("q")]

    def satisfied_dependencies(question):
        return [
            d for d in dependency_list(question) if answers.get(d["q"]) == d["value"]
        ]

    def realized_depth(question):
        satisfied = satisfied_dependencies(question)
        if not satisfied:
            return 0
        return 1 + min(realized_depth(by_id[d["q"]]) for d in satisfied)

    progressed = True
    while progressed:
        progressed = False
        for question in questions:
            qid = str(question["id"])
            if qid in rendered:
                continue
            deps = dependency_list(question)
            if deps and not satisfied_dependencies(question):
                continue
            answer = render_cascade_question(
                question["text"],
                question["alternatives"],
                key=f"resource_q_{qid}",
                level=realized_depth(question),
            )
            answers[qid] = answer
            rendered.add(qid)
            progressed = True
            result_entry = next(
                (r for r in question.get("result", []) if r["answer"] == answer),
                None,
            )
            if result_entry:
                render_recommended_mitigation_category(result_entry["action"])
    return answers


def resource_aware_section():
    questions = [
        {
            "id": 1,
            "text": "Do you have authorization and resources to run training?",
            "depends_on": {"q": "", "value": ""},
            "alternatives": ["yes", "no"],
        },
        {
            "id": 2,
            "text": "What level of access do you have on the model's output?",
            "depends_on": [{"q": "1", "value": "no"}, {"q": "5", "value": "no"}],
            "alternatives": [
                "scores_propabilities",
                "only_final_labels",
                "no_output_access",
            ],
            "result": [
                {"answer": "scores_propabilities", "action": "H"},
                {"answer": "no_output_access", "action": "J"},
            ],
        },
        {
            "id": 3,
            "text": "Can you add an external decision layer?",
            "depends_on": [
                {"q": "2", "value": "only_final_labels"},
                {"q": "9", "value": "None"},
            ],
            "alternatives": ["yes", "no"],
            "result": [
                {"answer": "yes", "action": "I"},
                {"answer": "no", "action": "J"},
            ],
        },
        {
            "id": 4,
            "text": "Can you access the training data?",
            "depends_on": {"q": "1", "value": "yes"},
            "alternatives": ["yes", "no"],
        },
        {
            "id": 5,
            "text": "Can you acquire or generate new data?",
            "depends_on": {"q": "4", "value": "no"},
            "alternatives": ["yes", "no"],
            "result": [
                {"answer": "yes", "action": "D"},
            ],
        },
        {
            "id": 6,
            "text": "Can you modify the dataset?",
            "depends_on": {"q": "4", "value": "yes"},
            "alternatives": ["yes", "no"],
        },
        {
            "id": 7,
            "text": "Can you change or engineer features?",
            "depends_on": {"q": "6", "value": "yes"},
            "alternatives": ["yes", "no"],
            "result": [
                {"answer": "yes", "action": "A"},
                {"answer": "no", "action": "B"},
            ],
        },
        {
            "id": 8,
            "text": "Can you reweight or resample instances?",
            "depends_on": {"q": "6", "value": "no"},
            "alternatives": ["yes", "no"],
            "result": [{"answer": "yes", "action": "C"}],
        },
        {
            "id": 9,
            "text": "What training scope is available?",
            "depends_on": {"q": "8", "value": "no"},
            "alternatives": ["full_retrain", "full_ft", "partial_ft", "None"],
            "result": [
                {"answer": "full_retrain", "action": "E"},
                {"answer": "full_ft", "action": "F"},
                {"answer": "partial_ft", "action": "G"},
            ],
        },
    ]
    st.write("Answer each question to reveal the next relevant one")
    answers = render_resource_aware_flow(questions)
    st.session_state["resource_aware_answers"] = answers
    return questions


def map_mitigation_recommendations():
    mitigation_categories = [
        {
            "category": "A",
            "description": "Data & Feature interventions",
            "group": "Pre-processing",
        },
        {
            "category": "B",
            "description": "Data-only interventions",
            "group": "Pre-processing",
        },
        {
            "category": "C",
            "description": "Reweighting or resampling",
            "group": "Pre-processing",
        },
        {
            "category": "D",
            "description": "Data Acquisition or generation",
            "group": "Pre-processing",
        },
        {
            "category": "E",
            "description": "Training from scratch with fairness objectives",
            "group": "In-processing",
        },
        {
            "category": "F",
            "description": "Full fine-tuning with fairness objectives",
            "group": "In-processing",
        },
        {
            "category": "G",
            "description": "Limited parameter tuning",
            "group": "In-processing",
        },
        {
            "category": "H",
            "description": "Output-only post-processing",
            "group": "Post-processing",
        },
        {
            "category": "I",
            "description": "Decision layer control",
            "group": "Post-processing",
        },
        {"category": "J", "description": "Governance only", "group": "Post-processing"},
    ]
    return mitigation_categories


# Renders whichever operation category was last clicked in the
# "AI system's stages and operations" grid (populate_stages() stashes it
# into st.session_state["selected_operation_implementation"]). There can be
# several aipc_*.yaml entries wired to the same operation type (e.g.
# baseline vs. fairness-aware model_training variants) -- one expander per
# entry.
def show_operation_implementation():
    selection = st.session_state.get("selected_operation_implementation")
    if not selection:
        st.info("Click an operation above to inspect its implementation.")
        return

    op_type = selection["op_type"]
    current_product = selection["current_product"]
    current_framework = selection["current_framework"]

    # Drafts added via the "Add new operation" button below, kept separate
    # from `selection["entries"]` (re-read from the aipc_*.yaml config each
    # time the operation card is clicked) so they survive reruns instead of
    # being wiped by that re-read.
    if "custom_operation_entries" not in st.session_state:
        st.session_state["custom_operation_entries"] = {}
    custom_entries = st.session_state["custom_operation_entries"].setdefault(op_type, [])
    implementations = selection["entries"] + custom_entries

    if not implementations:
        st.info(f"No wired implementation found for operation '{op_type}' yet.")
    else:
        report_artifacts = load_aipc_config(current_product, current_framework).get(
            "artifacts", {}
        ).get("report", [])

        for ind, entry in enumerate(implementations):
            specs = entry["implementation"]["spec"]
            method_name = specs["method_name"]
            step_operations_module = os.path.basename(specs["path"])
            framework = entry["implementation"].get("framework", current_framework)
            inputs = specs.get("inputs", [])
            outputs = specs.get("outputs", [])

            with st.expander(f"{entry['id']}: {entry.get('name', '')}", expanded=False):
                cols_oper = st.columns([7, 3])
                with cols_oper[0]:
                    tab1, tab2, tab3, tab4 = st.tabs(
                        [
                            "Code",
                            "Input",
                            "Output",
                            "Produced artifact",
                        ]
                    )
                    with tab1:
                        st.write("This is the Code tab")
                        run_method_name = method_name
                        run_module = step_operations_module
                        if framework == "dh":
                            run_method_name = method_name + "_real"
                            run_module = "dh_" + step_operations_module
                        try:
                            method_content = load_method_content(
                                run_method_name,
                                current_product,
                                framework,
                                run_module,
                            )
                        except Exception as exc:
                            method_content = (
                                f"# Could not load source for {run_method_name}: {exc}"
                            )
                        st_ace(
                            value=method_content,
                            language="python",
                            theme="xcode",
                            key=f"code_{op_type}_{ind}",
                            height=300,
                            font_size=14,
                            show_gutter=True,
                            readonly=False,
                        )
                    with tab2:
                        st.write("This tab contains Input data that the methods receives")
                        st_ace(
                            value=json.dumps(inputs, indent=2),
                            language="json",
                            theme="xcode",
                            key=f"input_{op_type}_{ind}",
                            height=300,
                            font_size=14,
                            show_gutter=True,
                            readonly=False,
                        )
                    with tab3:
                        st.write("This tab contains Output data that the methods produces")
                        st_ace(
                            value=json.dumps(outputs, indent=2),
                            language="json",
                            theme="xcode",
                            key=f"output_{op_type}_{ind}",
                            height=300,
                            font_size=14,
                            show_gutter=True,
                            readonly=False,
                        )
                    with tab4:
                        import streamlit.components.v1 as components

                        shown_any = False
                        for output in outputs:
                            if "report" not in output:
                                continue
                            report = next(
                                (
                                    r
                                    for r in report_artifacts
                                    if r["name"] == output["report"]
                                ),
                                None,
                            )
                            if not report:
                                continue
                            report_path = os.path.join(
                                USE_CASES_FOLDER,
                                current_product,
                                "src",
                                f"{framework}_platform",
                                "artifacts",
                                "report",
                                report["config"]["filepath"],
                            )
                            if report_path.endswith("html") and os.path.exists(
                                report_path
                            ):
                                shown_any = True
                                with open(report_path, encoding="utf8") as report_f:
                                    components.html(
                                        report_f.read(),
                                        width=1000,
                                        height=1200,
                                        scrolling=True,
                                    )
                        if not shown_any:
                            st.info("No produced artifact preview available yet.")
                with cols_oper[1]:
                        st.write(
                            "The following code is a recommended implementation for this operation, derived from a catalog of open-source tools."
                        )
                        tools_catalog = pd.read_csv(
                            os.path.join(
                                TOOLS_CATALOG_FOLDER, "tools_principles_catalog.csv"
                            )
                        )
                        code_catalog = tools_catalog[
                            tools_catalog["ai_operation"].isin([entry["type"]])
                        ]
                        print(code_catalog)
                        for tool in code_catalog.itertuples():
                            snippet_path = tool.code_snippet_path
                            toolname = tool.tool
                            documentation = tool.documentation
                            file = os.path.join(
                                TOOLS_CATALOG_FOLDER, snippet_path.split(":")[0]
                            )
                            method = snippet_path.split(":")[1]
                            method_snippet = get_function_source(file, method)
                            with st.expander(
                                f"Tool suggestion: {toolname}", expanded=False
                            ):
                                st.markdown(
                                    f"""
                                        <a href="{documentation}" target="_blank">Tool documentation ↗</a>
                                    """,
                                    unsafe_allow_html=True,
                                )
                                st_ace(
                                    value=method_snippet,
                                    language="json",
                                    theme="xcode",
                                    key=f"suggestion_{ind}_{tool.Index}",
                                    height=300,
                                    font_size=14,
                                    show_gutter=True,
                                    readonly=False,
                                )

                if st.button("Run operation", key=f"run_op_{op_type}_{ind}"):
                    pass

    btn_col, _ = st.columns([3, 9])
    with btn_col:
        if st.button("➕ Add new operation", key=f"add_op_{op_type}"):
            custom_entries.append(
                {
                    "id": f"draft-{len(custom_entries) + 1}",
                    "name": "New operation",
                    "type": op_type,
                    "implementation": {
                        "framework": current_framework,
                        "spec": {
                            "method_name": "",
                            "path": "",
                            "inputs": [],
                            "outputs": [],
                        },
                    },
                }
            )
            st.rerun()


# Step 2: definition of the requirements dimensions to be satisfied according to the AI product design objectives
def show_new_prod_requirements():
    st.session_state["page"] = "new_prod"
    render_section_header("AI system's requirements dimensions")
    checklist_requirements = st.multiselect(
        "AI system's requirements dimensions",
        requirements_dimensions,
        ["Baseline", "Robustness"],
        label_visibility="collapsed",
    )


# Step 3: specification of the data artifacts and the AI product objectives (classification, clustering, information extraction)
# subtasks:
#   - load data artifact from corresponding dh project
#   - load data characteristics into the prompt context in order to facilitate the planning of the operations
# data_atifact = pd.read_csv("artifacts/data/data.csv") # TODO


# Step 4: CoT prompting to plan the operations of a new AI product
# subtasks:
#   - select the right open source toolkits to use for implementing each operation according to the pre-defined requirements
#   - generate code snippets based on the selected toolkits for each operation
async def run_mcp_query(user_input):
    # Model
    model = ChatOpenAI(model="gpt-5")  # "gpt-5"

    # MCP Client via HTTP
    client = MultiServerMCPClient(
        {
            "mlops_tai_engineers": {
                "transport": "streamable_http",
                "url": "http://127.0.0.1:8082/mcp",
            },
            "file_system": {
                "transport": "streamable_http",
                "url": "http://127.0.0.1:8080/mcp",
            },
        }
    )
    tools = await client.get_tools()  # await load_mcp_tools(client) #
    resources = await client.get_resources("mlops_tai_engineers")
    model_with_tools = model.bind_tools(tools)
    tool_node = ToolNode(tools)

    def should_continue(state: MessagesState):

        messages = state["messages"]
        last_message = messages[-1]
        if last_message.tool_calls:
            return "tools"
        return END

    async def call_model(state: MessagesState):
        messages = state["messages"]
        response = await model_with_tools.ainvoke(messages)
        return {"messages": [response]}

    # LangGraph pipeline
    builder = StateGraph(MessagesState)
    builder.add_node("call_model", call_model)
    builder.add_node("tools", tool_node)
    builder.add_edge(START, "call_model")
    builder.add_conditional_edges("call_model", should_continue)
    builder.add_edge("tools", "call_model")

    graph = builder.compile()
    result = await graph.ainvoke(
        {"messages": [{"role": "user", "content": user_input}]}
    )

    # Extract last message text
    last_msg = result["messages"][-1].content
    return last_msg if isinstance(last_msg, str) else str(last_msg)


# Step 5: generate the new AI product folder structure with the necessary code files and configuration files
def generate_prod_action(ai_prod_desc):
    if st.button("Create AI product skeleton", key=f"generate_product"):
        template_aipc_folder = os.path.join(
            parent_folder, "framework/temlops/aipc_template"
        )
        new_prod_folder = os.path.join(
            parent_folder, "framework/temlops/use_cases/new_prod"
        )
        folder = f" FOLDER: {new_prod_folder}"
        os.makedirs(new_prod_folder, exist_ok=True)
        with st.spinner("Thinking..."):
            selected_ops = st.session_state["selected_operations"]
            selected_operations = [
                op for op, selected in selected_ops.items() if selected
            ]
            plan_prompt = open(f"guided_ui/pages/plan.md", "r").read()
            ai_prod_desc = f"{plan_prompt}. \n\n Copy recursively the files and subdirectories inside the folder {template_aipc_folder} into the new AI product folder {new_prod_folder}.  \n\n  The selected operations are: {selected_operations}"
            answer = asyncio.run(run_mcp_query(ai_prod_desc))
            st.success("Operation completed successfully!")
            st.success(answer)


if __name__ == "__main__":
    ai_prod_name, ai_prod_desc = fairness_settings()
    with st.container(border=True):
        tab1, tab2 = st.tabs(
            [
                "1.Resource-aware selection flow for bias mitigation",
                "2.Fairness Concerns",
            ]
        )
        with tab1:
            resource_aware_section()
        with tab2:
            fairness_requirements_section()
    with st.container(border=True):
        lifecycle_stages()
        show_operation_implementation()
    with st.container(border=True):
        show_new_prod_requirements()
    generate_prod_action(ai_prod_desc)
