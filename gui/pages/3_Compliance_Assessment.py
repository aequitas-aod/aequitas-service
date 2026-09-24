# streamlit: page_name = "Compliance Assessment"
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

st.set_page_config(layout="wide", page_title="Compliance Assessment")  # , page_icon="📊"
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
        <a class="nav-link" href="#fairness-settings">→ Fairness settings</a>
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


def fairness_settings():
    render_section_header("Fairness settings", anchor="fairness-settings")
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


def lifecycle_stages():
    render_section_header(
        "AI system's stages and operations", anchor="lifecycle-stages"
    )
    recommended_group = st.session_state.get("recommended_mitigation_group")
    populate_stages(
        pipeline_configs, createview=True, recommended_group=recommended_group
    )


# Renders a single Fairness Notion's expander: concern badges, its Fairness
# Metrics and the concern's Mitigation Techniques (via
# MITIGATION_TECHNIQUE_FOR_CONCERN_QUERY) side by side, plus the "add custom
# metric" affordance. Appends any checked metric/technique IRIs into the
# caller's `selected_metrics` / `selected_mitigation_techniques` lists.
def render_fairness_notion_expander(
    concern, notion, notion_metrics, concerns_notions, selected_metrics, selected_mitigation_techniques
):
    artifacts = load_aipc_config(DEFAULT_PRODUCT, DEFAULT_FRAMEWORK).get("artifacts", {})

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

        metrics_col, mitigation_col = st.columns(2, gap="small", border=True)
        with metrics_col:
            st.markdown(
                "<div style='font-size:13px; font-weight:700; color:#374151; "
                "text-transform:uppercase; letter-spacing:0.03em; "
                "border-bottom:2px solid #e5e7eb; padding-bottom:6px; "
                "margin-bottom:10px;'>Fairness Metrics</div>",
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
                    # Reuses render_operation_entry_expander() -- the same
                    # Code/Input/Output/Produced artifact + tool suggestions
                    # expander shown for a wired operation -- so a selected
                    # metric's implementation can be reviewed/edited right
                    # under its checkbox.
                    render_operation_entry_expander(
                        {
                            "id": "Implementation",
                            "name": metric["label"],
                            "type": "model_evaluation",
                            "implementation": {
                                "framework": DEFAULT_FRAMEWORK,
                                "spec": FAIRNESS_METRIC_REFERENCE_SPEC,
                            },
                        },
                        0,
                        DEFAULT_PRODUCT,
                        DEFAULT_FRAMEWORK,
                        artifacts,
                        key_prefix=f"metric_impl_{concern['iri']}_{notion['iri']}_{metric['iri']}",
                        show_tool_suggestions=False,
                    )

        with mitigation_col:
            st.markdown(
                "<div style='font-size:13px; font-weight:700; color:#374151; "
                "text-transform:uppercase; letter-spacing:0.03em; "
                "border-bottom:2px solid #e5e7eb; padding-bottom:6px; "
                "margin-bottom:10px;'>Bias Mitigation techniques</div>",
                unsafe_allow_html=True,
            )
            for technique in get_mitigation_techniques_for_concern(concern["iri"]):
                technique_key = f"technique_{concern['iri']}_{notion['iri']}_{technique['iri']}"
                technique_checked = render_cascade_checkbox(
                    technique["label"],
                    key=technique_key,
                    level=0,
                )
                if technique_checked:
                    selected_mitigation_techniques.append(technique["iri"])
                    # Reuses render_operation_entry_expander() -- the same
                    # Code/Input/Output/Produced artifact + tool suggestions
                    # expander shown for a wired operation -- so a selected
                    # mitigation technique's implementation can be
                    # reviewed/edited right under its checkbox.
                    render_operation_entry_expander(
                        {
                            "id": "Implementation",
                            "name": technique["label"],
                            "type": "model_training",
                            "implementation": {
                                "framework": DEFAULT_FRAMEWORK,
                                "spec": MITIGATION_TECHNIQUE_REFERENCE_SPEC,
                            },
                        },
                        0,
                        DEFAULT_PRODUCT,
                        DEFAULT_FRAMEWORK,
                        artifacts,
                        key_prefix=f"technique_impl_{concern['iri']}_{notion['iri']}_{technique['iri']}",
                        show_tool_suggestions=False,
                    )

        notion_key = str(notion["iri"]).split("#")[-1]

        # Custom metric drafts reuse render_operation_entry_expander() --
        # the exact same Code/Input/Output/Produced artifact + tool
        # suggestions expander shown when clicking an operation category in
        # "AI system's stages and operations" -- so a custom metric is
        # authored with the same UI as a wired operation.
        if "custom_metric_entries" not in st.session_state:
            st.session_state["custom_metric_entries"] = {}
        custom_metric_entries = st.session_state["custom_metric_entries"].setdefault(
            notion_key, []
        )

        for ind, metric_entry in enumerate(custom_metric_entries):
            render_operation_entry_expander(
                metric_entry,
                ind,
                DEFAULT_PRODUCT,
                DEFAULT_FRAMEWORK,
                artifacts,
                key_prefix=f"custom_metric_{notion_key}",
            )

        btn_col, _ = st.columns([3, 9])
        with btn_col:
            if st.button(
                "➕ Add custom metric",
                key=f"add_custom_metric_btn_{notion_key}",
            ):
                custom_metric_entries.append(
                    {
                        "id": f"draft-{len(custom_metric_entries) + 1}",
                        "name": "New metric",
                        "type": "custom_metric",
                        "implementation": {
                            "framework": DEFAULT_FRAMEWORK,
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


# Step 1b: cascading fairness requirements for the selected AI Type of Use.
# Checking a Fairness Concern reveals its Fairness Notions, grouped into the
# three mutually exclusive group-fairness criteria from the fairness
# impossibility theorem -- Independence, Separation, Sufficiency (per
# get_fairness_notion_category) -- plus an "Other notions" tab for notions
# outside that trilemma (e.g. individual/causal/procedural fairness
# notions), so nothing found in the ontology is left unshown.
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

    tab_names = list(FAIRNESS_NOTION_CATEGORIES) + ["Other notions"]
    notions_by_category = {name: [] for name in tab_names}

    for concern in concerns:
        for notion in get_fairness_notions(concern["iri"]):
            notion_metrics = get_fairness_metrics(notion["iri"])
            if notion["label"] not in concerns_notions_vis and len(notion_metrics) > 0:
                concerns_notions_vis[notion["label"]].append(
                    str(concern["iri"]).split("#")[-1]
                )  # notion
                category = get_fairness_notion_category(notion["iri"]) or "Other notions"
                notions_by_category[category].append((concern, notion, notion_metrics))

    for tab, category in zip(st.tabs(tab_names), tab_names):
        with tab:
            entries = notions_by_category[category]
            if not entries:
                st.info(f"No {category.lower()} found for this AI Type of Use.")
                continue
            for concern, notion, notion_metrics in entries:
                render_fairness_notion_expander(
                    concern,
                    notion,
                    notion_metrics,
                    concerns_notions,
                    selected_metrics,
                    selected_mitigation_techniques,
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


if __name__ == "__main__":
    render_sidebar_navigation()

    ai_prod_name, ai_prod_desc = ai_product_header()

    with st.container(border=True):
        artifacts_overview_section()

    fairness_settings()
    with st.container(border=True):
        tab1, tab2, tab3 = st.tabs(
            [
                "1.Resource-aware selection flow for bias mitigation",
                "2.Questions",
                "3.Fairness Concerns",
            ]
        )
        with tab1:
            resource_aware_section()
        with tab2:
            render_section_header("Questions")
            render_competency_questions()
        with tab3:
            fairness_requirements_section()
        
    with st.container(border=True):
        lifecycle_stages()
        show_operation_implementation()
