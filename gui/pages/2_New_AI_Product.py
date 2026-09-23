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
    get_fairness_notion_category,
    FAIRNESS_NOTION_CATEGORIES,
    get_fairness_metrics,
    get_mitigation_techniques_for_concern,
    render_cascade_checkbox,
    render_cascade_question,
    render_competency_questions,
    load_method_content,
    load_aipc_config,
    import_from_path,
    get_function_source,
    run_operation,
)
from dotenv import load_dotenv, find_dotenv

from langchain_mcp_adapters.client import MultiServerMCPClient
from langchain_mcp_adapters.tools import load_mcp_tools
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, MessagesState, START, END
from langgraph.prebuilt import ToolNode

_ = load_dotenv(find_dotenv())

st.set_page_config(layout="wide", page_title="New AI Product")  # , page_icon="📊"
st.title("New AI Product")
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


def render_section_header(text, anchor=None):
    """Section title styled larger than the default st.write() body text.
    `anchor`, if given, sets an element id so the sidebar navigation links
    (see render_sidebar_navigation()) can jump to it via "#anchor"."""
    anchor_attr = f' id="{anchor}"' if anchor else ""
    st.markdown(
        f"<p{anchor_attr} style='font-size:22px; font-weight:600;'>{text}</p>",
        unsafe_allow_html=True,
    )


def render_sidebar_navigation():
    """Sidebar links that jump to the "Artifacts overview", "Fairness
    settings" and "Lifecycle stages" sections of the page via in-page anchors
    set through render_section_header()'s `anchor` param."""
    st.sidebar.markdown(
        """
        <style>
        .nav-section-title {
            font-size: 12px;
            font-weight: 700;
            color: #6b7280;
            text-transform: uppercase;
            letter-spacing: 0.06em;
            margin: 4px 0 10px 2px;
        }
        .nav-link {
            display: flex;
            align-items: center;
            gap: 8px;
            padding: 10px 12px;
            margin-bottom: 8px;
            border-radius: 8px;
            background-color: #f3f4f6;
            border-left: 3px solid transparent;
            color: #1f2937 !important;
            font-weight: 600;
            font-size: 14px;
            text-decoration: none !important;
            transition: background-color 0.15s ease, border-left-color 0.15s ease;
        }
        .nav-link:hover {
            background-color: #e0e7ff;
            border-left-color: #4f46e5;
            color: #3730a3 !important;
        }
        </style>
        <div class="nav-section-title">Sections</div>
        <a class="nav-link" href="#artifacts-overview">→ Artifacts overview</a>
        <a class="nav-link" href="#lifecycle-stages">→ Lifecycle stages</a>
        """,
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

# Fallback product/framework for content not tied to a specific wired
# operation selection (e.g. a custom fairness metric draft) -- mirrors
# populate_stages()'s own defaults.
DEFAULT_PRODUCT = "recruitment"
DEFAULT_FRAMEWORK = "local"

# Reference implementation shown in a selected fairness metric's expander --
# model_evaluation_fairness_disparate_impact_remover in the recruitment
# use case's local_platform (wired under this same product/framework in
# aipc_local.yaml), so its real source loads in the Code tab and its actual
# artifact specs populate Input/Output, making it runnable as-is.
FAIRNESS_METRIC_REFERENCE_SPEC = {
    "path": "modelling.py",
    "method_name": "model_evaluation_fairness_disparate_impact_remover",
    "inputs": [
        {"data_test": "data_testing"},
        {"model": "model_fairness_disparate_impact_remover"},
        {"config": "model_evaluation_fairness_metrics"},
    ],
    "outputs": [
        {"report": "model_fairness_metrics_report"},
    ],
}

# Reference implementation shown in a selected mitigation technique's
# expander -- train_model_disparate_impact_remover in the recruitment use
# case's local_platform (wired under this same product/framework in
# aipc_local.yaml), so its real source loads in the Code tab and its actual
# artifact specs populate Input/Output, making it runnable as-is.
MITIGATION_TECHNIQUE_REFERENCE_SPEC = {
    "path": "modelling.py",
    "method_name": "train_model_disparate_impact_remover",
    "inputs": [
        {"data": "data_training"},
        {"config": "model_train_fair"},
    ],
    "outputs": [
        #{"model": "model_fairness_disparate_impact_remover"},
        {"report": "model_performance_metrics_report"}
    ],
}


# Step 1: definition of the necessary operations (active/inactive ops)
with open(pipeline_definitions_folder, "r") as yaml_file:
    pipeline_configs = yaml.safe_load(yaml_file)
    requirements_dimensions = list(
        map(lambda x: x.capitalize(), pipeline_configs["requirements_dimensions"])
    )


# Where each artifacts:<category> entry's file (if any) lives on disk --
# mirrors the *_ARTIFACTS_PATH constants in the use case's
# local_platform/platform_artifacts.py. Configuration, Function and Service
# (temlops/src/artifact_types.py) aren't file-backed -- their entries are
# declarative parameter blobs with no consistent file field -- so they're
# left out and only get their raw definition rendered.
ARTIFACT_CATEGORY_SUBFOLDER = {
    "data": "data",
    "model": "model",
    "report": "report",
    "status": "status",
}

FILE_BACKED_ARTIFACT_CATEGORIES = set(ARTIFACT_CATEGORY_SUBFOLDER) | {"documentation"}

# Categories whose file, when present on disk, is rendered as an inline
# preview rather than just a "file on disk" caption -- the same csv/image/html
# formats already handled for produced reports in
# render_operation_entry_expander().
PREVIEWABLE_ARTIFACT_CATEGORIES = {"data", "report"}


def _artifact_relative_path(category, artifact):
    """The filepath-like field of one artifacts:<category> entry, wherever it
    sits in that entry's shape: `status` entries carry `status_file` at the
    top level, `data`/`model`/`report` nest `filepath` under `config`,
    `documentation` entries carry a top-level `filepath`."""
    field = "status_file" if category == "status" else "filepath"
    if field in artifact:
        return artifact[field]
    config = artifact.get("config")
    return config.get(field) if isinstance(config, dict) else None


def _artifact_file_path(category, artifact, current_product, current_framework):
    if category not in FILE_BACKED_ARTIFACT_CATEGORIES:
        return None
    relative_path = _artifact_relative_path(category, artifact)
    if not relative_path:
        return None
    if category in ARTIFACT_CATEGORY_SUBFOLDER:
        return os.path.join(
            USE_CASES_FOLDER,
            current_product,
            "src",
            f"{current_framework}_platform",
            "artifacts",
            ARTIFACT_CATEGORY_SUBFOLDER[category],
            relative_path,
        )
    # `documentation` entries (e.g. "cards/data_card.md") are stored relative
    # to the use case root, not the artifacts/ subtree.
    return os.path.join(USE_CASES_FOLDER, current_product, relative_path)


def _render_artifact_preview(file_path):
    """Best-effort inline preview of an artifact's file: csv as a table,
    images inline, html embedded -- the exact formats already handled for
    produced reports in render_operation_entry_expander(). Returns whether
    anything was shown."""
    lower_path = file_path.lower()
    if lower_path.endswith(".csv"):
        st.dataframe(pd.read_csv(file_path), use_container_width=True)
        return True
    if lower_path.endswith((".png", ".jpg", ".jpeg", ".gif", ".bmp", ".webp")):
        st.image(file_path, caption=os.path.basename(file_path), use_container_width=True)
        return True
    if lower_path.endswith(".html"):
        import streamlit.components.v1 as components

        with open(file_path, encoding="utf8") as report_f:
            components.html(report_f.read(), width=1000, height=600, scrolling=True)
        return True
    return False


def _render_artifact_definition(artifact, key):
    """Renders one artifacts:<category> entry's declarative specification --
    the exact YAML block defining it in aipc_<framework>.yaml -- inside a
    read-only textarea."""
    spec_yaml = yaml.dump(artifact, sort_keys=False, default_flow_style=False)
    st.text_area(
        "Declarative specification",
        value=spec_yaml,
        key=key,
        height=150,
        disabled=True,
    )


def _render_artifact_entry(category, artifact, current_product, current_framework, key):
    with st.expander(artifact.get("name", "(unnamed)"), expanded=False):
        _render_artifact_definition(artifact, key=f"artifact_spec_{key}")
        file_path = _artifact_file_path(category, artifact, current_product, current_framework)
        if not file_path:
            return
        if not os.path.exists(file_path):
            st.info("Not generated yet — no file found on disk for this artifact.")
            return
        if category in PREVIEWABLE_ARTIFACT_CATEGORIES and _render_artifact_preview(file_path):
            return
        st.caption(f"File on disk: {file_path}")


def artifacts_overview_section():
    """Renders the `artifacts:` key of aipc_<framework>.yaml as one tab per
    category (data, model, configuration, report, status, documentation,
    function, service), each entry in its own collapsed expander showing its
    raw definition plus, for data/report entries whose file exists on disk,
    an inline preview."""
    render_section_header("Artifacts overview", anchor="artifacts-overview")
    artifacts = load_aipc_config(DEFAULT_PRODUCT, DEFAULT_FRAMEWORK).get("artifacts", {})
    if not artifacts:
        st.info("No artifacts defined yet for this AI product.")
        return

    categories = list(artifacts.keys())
    tab_labels = [category.replace("_", " ").capitalize() for category in categories]
    for tab, category in zip(st.tabs(tab_labels), categories):
        with tab:
            entries = artifacts[category]
            if not entries:
                st.info(f"No {category} artifacts defined.")
                continue
            for ind, artifact in enumerate(entries):
                _render_artifact_entry(
                    category,
                    artifact,
                    DEFAULT_PRODUCT,
                    DEFAULT_FRAMEWORK,
                    key=f"{category}_{ind}",
                )


def ai_product_header():
    """AI product name/description textareas -- shown once at the top of the
    page since they identify the product as a whole, rather than being
    specific to the fairness settings below them."""
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
    return ai_prod_name, ai_prod_desc

def lifecycle_stages():
    render_section_header(
        "AI system's stages and operations", anchor="lifecycle-stages"
    )
    recommended_group = st.session_state.get("recommended_mitigation_group")
    populate_stages(
        pipeline_configs, createview=True, recommended_group=recommended_group
    )

# Renders one aipc_*.yaml operation entry (or a synthetic entry with the
# same shape, e.g. an "Add new operation" / "Add custom metric" draft) as an
# expander: a Code/Input/Output/Produced artifact tab group, optionally next
# to a column of tool-catalog suggestions matching the entry's `type`, plus
# a "Run operation" button. `key_prefix` namespaces widget keys per call
# site (an operation type, or a fairness notion for custom-metric drafts) so
# the same renderer can be reused verbatim from multiple places.
# `show_tool_suggestions` hides the recommended-implementation column for
# call sites where the entry isn't a real pipeline operation (e.g. a
# fairness metric or mitigation technique checkbox) and the tools catalog
# -- keyed by pipeline operation type -- has nothing relevant to suggest.
def render_operation_entry_expander(
    entry,
    ind,
    current_product,
    current_framework,
    artifacts,
    key_prefix,
    show_tool_suggestions=True,
):
    specs = entry["implementation"]["spec"]
    method_name = specs["method_name"]
    step_operations_module = os.path.basename(specs["path"])
    framework = entry["implementation"].get("framework", current_framework)
    inputs = specs.get("inputs", [])
    outputs = specs.get("outputs", [])
    report_artifacts = artifacts.get("report", [])
    data_artifacts = artifacts.get("data", [])
    model_artifacts = artifacts.get("model", [])
    configuration_artifacts = artifacts.get("configuration", [])

    with st.expander(f"{entry['id']}: {entry.get('name', '')}", expanded=False):
        if show_tool_suggestions:
            cols_oper = st.columns([7, 3])
            code_col = cols_oper[0]
        else:
            code_col = st.container()
        with code_col:
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
                    key=f"code_{key_prefix}_{ind}",
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
                    key=f"input_{key_prefix}_{ind}",
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
                    key=f"output_{key_prefix}_{ind}",
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
                    if not os.path.exists(report_path):
                        continue
                    if report_path.endswith("html"):
                        shown_any = True
                        with open(report_path, encoding="utf8") as report_f:
                            components.html(
                                report_f.read(),
                                width=1000,
                                height=1200,
                                scrolling=True,
                            )
                    elif report_path.endswith("csv"):
                        shown_any = True
                        st.dataframe(
                            pd.read_csv(report_path), use_container_width=True
                        )
                    elif report_path.lower().endswith(
                        (".png", ".jpg", ".jpeg", ".gif", ".bmp", ".webp")
                    ):
                        shown_any = True
                        st.image(
                            report_path,
                            caption=os.path.basename(report_path),
                            use_container_width=True,
                        )
                if not shown_any:
                    st.info("No produced artifact preview available yet.")
        if show_tool_suggestions:
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
                            key=f"suggestion_{key_prefix}_{ind}_{tool.Index}",
                            height=300,
                            font_size=14,
                            show_gutter=True,
                            readonly=False,
                        )

        if st.button("Run operation ", key=f"run_op_{key_prefix}_{ind}"):
            try:
                run_operation(
                    entry,
                    data_artifacts,
                    model_artifacts,
                    configuration_artifacts,
                    report_artifacts,
                    current_product,
                    framework,
                    run_module,
                )
                st.success("Operation completed successfully!")
            except Exception as exc:
                st.error(f"Could not run operation: {exc}")
              
        if not show_tool_suggestions:
            if st.button("Select operation ", key=f"select_op_{key_prefix}_{ind}"):
                pass
                # 


# Renders whichever operation category was last clicked in the
# "AI system's stages and operations" grid (populate_stages() stashes it
# into st.session_state["selected_operation_implementation"]). There can be
# several aipc_*.yaml entries wired to the same operation type (e.g.
# baseline vs. fairness-aware model_training variants) -- one expander per
# entry, via render_operation_entry_expander().
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
        artifacts = load_aipc_config(current_product, current_framework).get(
            "artifacts", {}
        )

        for ind, entry in enumerate(implementations):
            render_operation_entry_expander(
                entry, ind, current_product, current_framework, artifacts, key_prefix=op_type
            )

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

def generate_operation_operation_code(operation_category):
    if st.button("Generate operation code", key=f"generate_operation_code"):
        with st.spinner("Thinking..."):
            selected_ops = st.session_state["selected_operations"]
            selected_operations = [
                op for op, selected in selected_ops.items() if selected
            ]
            plan_prompt = open(f"guided_ui/pages/plan.md", "r").read()
            ai_prod_desc = f"{plan_prompt}. \n\n The selected operations are: {selected_operations}"
            answer = asyncio.run(run_mcp_query(ai_prod_desc))
            st.success("Operation completed successfully!")
            st.success(answer)

if __name__ == "__main__":
    render_sidebar_navigation()

    ai_prod_name, ai_prod_desc = ai_product_header()

    with st.container(border=True):
        artifacts_overview_section()
        
    with st.container(border=True):
        lifecycle_stages()
        show_operation_implementation()
    with st.container(border=True):
        show_new_prod_requirements()
    generate_prod_action(ai_prod_desc)
