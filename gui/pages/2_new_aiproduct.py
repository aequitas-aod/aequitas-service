# streamlit: page_name = "New AI Product"
import os
import yaml
import asyncio
import pandas as pd
import streamlit as st
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
    get_mitigation_techniques,
    render_cascade_checkbox,
    render_cascade_question,
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
    </style>
    """,
    unsafe_allow_html=True,
)


def render_section_header(text):
    """Section title styled larger than the default st.write() body text."""
    st.markdown(f"<p style='font-size:22px; font-weight:600;'>{text}</p>", unsafe_allow_html=True)

current_folder = os.path.dirname(os.path.abspath(__file__))
parent_folder = os.path.dirname(current_folder)
parent_folder = os.path.dirname(parent_folder)
pipeline_definitions_folder = os.path.join(
    parent_folder, "framework/temlops/config/pipeline_definitions.yaml"
)

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
    return ai_prod_name, ai_prod_desc


def lifecycle_stages():
    render_section_header("AI system's stages and operations")
    selected_operations = populate_stages(pipeline_configs, createview=True)
    st.session_state["selected_operations"] = selected_operations


# Step 1b: cascading fairness requirements for the selected AI Type of Use.
# Checking a Fairness Concern reveals its Fairness Notions; checking a Notion
# reveals the Fairness Metrics that measure it; checking a Metric reveals the
# Mitigation Techniques that enforce it 
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
            concerns_notions[notion['label']].append(concern["label"]) #notion

    for concern in concerns:
        #with st.expander(
        #    f"{concern['label']}. {concern['definition']}", expanded=False
        #):
        
        for notion in get_fairness_notions(concern["iri"]):
            if notion["label"] not in concerns_notions_vis:
                concerns_notions_vis[notion['label']].append(str(concern["iri"]).split("#")[-1]) #notion
            
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
                    for metric in get_fairness_metrics(notion["iri"]):
                        metric_key = (
                            f"metric_{concern['iri']}_{notion['iri']}_{metric['iri']}"
                        )
                        metric_checked = render_cascade_checkbox(
                            metric["label"],
                            key=metric_key,
                            level=2,
                        )
                        if metric_checked:
                            selected_metrics.append(metric["iri"])
                            for technique in get_mitigation_techniques(metric["iri"]):
                                technique_checked = render_cascade_checkbox(
                                    technique["label"],
                                    key=f"{metric_key}_{technique['iri']}",
                                    level=3,
                                )
                                if technique_checked:
                                    selected_mitigation_techniques.append(
                                        technique["iri"]
                                    )
    st.session_state["selected_fairness_metrics"] = selected_metrics
    st.session_state["selected_mitigation_techniques"] = (
        selected_mitigation_techniques
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
                st.caption(
                    f"↳ Recommended mitigation category: **{result_entry['action']}**"
                )
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
    print(tools)
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
            print(ai_prod_desc)
            answer = asyncio.run(run_mcp_query(ai_prod_desc))
            st.success("Operation completed successfully!")
            st.success(answer)


if __name__ == "__main__":
    ai_prod_name, ai_prod_desc = fairness_settings()
    tab1, tab2 = st.tabs(
        ["1.Resource-aware selection flow for bias mitigation", "2.Fairness Concerns"]
    )
    with tab1:
        resource_aware_section()
    with tab2:
        fairness_requirements_section()
    lifecycle_stages()
    show_new_prod_requirements()
    generate_prod_action(ai_prod_desc)
