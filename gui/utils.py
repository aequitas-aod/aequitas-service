import os
import ast
import sys
import re
from dotenv import load_dotenv, find_dotenv
import yaml
import pandas as pd
import streamlit as st
from rdflib import Graph
import importlib.util
import importlib
import inspect

load_dotenv(find_dotenv())

current_folder = os.path.dirname(os.path.abspath(__file__))
parent_folder = os.path.dirname(current_folder)
pipeline_definitions_folder = os.path.join(
    parent_folder, "framework/temlops/config/pipeline_definitions.yaml"
)
USE_CASES_FOLDER = os.path.join(parent_folder, "framework/temlops/use_cases")

# Path to the fairops ontology (lives in the separate fairness_ontology/fairops
# project). Override with the FAIROPS_ONTOLOGY_PATH env var if it's located
# elsewhere on disk.
FAIROPS_ONTOLOGY_PATH = os.environ.get(
    "FAIROPS_ONTOLOGY_PATH",
)
print(FAIROPS_ONTOLOGY_PATH)
# indiv.ttl holds the FairnessNotion/FairnessMetric individuals (under the
# indiv: namespace) and always ships alongside fairops.ttl in the same docs
# folder, so it's derived rather than configured separately.
FAIROPS_INDIVIDUALS_PATH = os.path.join(
    os.path.dirname(FAIROPS_ONTOLOGY_PATH), "indiv.ttl"
)

APPLICATION_DOMAIN_QUERY = """
PREFIX rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#>
PREFIX core: <https://purl.org/fairops/core#>

SELECT ?individual
WHERE {
    ?individual rdf:type core:ApplicationDomain .
}
"""

AI_TASK_QUERY = """
PREFIX rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#>
PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
PREFIX core: <https://purl.org/fairops/core#>

SELECT ?individual
WHERE {
    ?individual rdf:type ?type .
    ?type rdfs:subClassOf* core:AITask .
}
"""

AI_TYPE_OF_USE_QUERY = """
PREFIX rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#>
PREFIX core: <https://purl.org/fairops/core#>

SELECT ?individual
WHERE {
    ?individual rdf:type core:AITypeOfUse .
}
"""

# {ai_type_of_use_iri} is substituted with a full IRI wrapped in <> so the
# query works regardless of which prefix (core:/indiv:) the individual uses.
FAIRNESS_CONCERN_QUERY = """
PREFIX rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#>
PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
PREFIX core: <https://purl.org/fairops/core#>
PREFIX skos: <http://www.w3.org/2004/02/skos/core#>

SELECT DISTINCT ?fc ?def
WHERE {{
    ?fc rdf:type ?type .
    ?type rdfs:subClassOf* core:FairnessConcern .
    ?fc core:arisesIn <{ai_type_of_use_iri}> .
    OPTIONAL {{ ?fc skos:definition ?def }}
}}
"""

FAIRNESS_NOTION_QUERY = """
PREFIX core: <https://purl.org/fairops/core#>

SELECT DISTINCT ?notion
WHERE {{
    <{concern_iri}> core:isAddressedWith ?notion .
}}
"""

# Uses the permissive `measures` relation (not the SWRL-derived
# `isQuantifiedBy`, which needs a reasoning pass that hasn't been run against
# this ontology) per the "if we are just interested in ALL the metrics" note
# in sparql_toolQ.md.
FAIRNESS_METRIC_QUERY = """
PREFIX core: <https://purl.org/fairops/core#>

SELECT DISTINCT ?fm
WHERE {{
    ?fm core:measures <{notion_iri}> .
}}
"""

# Uses the permissive `enforces` relation (not the SWRL-derived
# `mitigatedWith`, which needs a reasoning pass that hasn't been run against
# this ontology) mirroring the `measures`/`isQuantifiedBy` choice made for
# FAIRNESS_METRIC_QUERY above. See "Retriving relevant mitigation
# techniques" in sparql_toolQ.md.
MITIGATION_TECHNIQUE_CATEGOERY_QUERY = """
PREFIX rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#>
PREFIX owl: <http://www.w3.org/2002/07/owl#>
PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
PREFIX xsd: <http://www.w3.org/2001/XMLSchema#>
PREFIX core: <https://purl.org/fairops/core#>
PREFIX skos: <http://www.w3.org/2004/02/skos/core#>
PREFIX indiv: <https://purl.org/fairops/indiv#>

SELECT ?individual ?subclass
WHERE {
    ?individual rdf:type ?subclass .
    ?subclass rdfs:subClassOf* core:DataGeneration .
}
"""

# Mirrors the "Retriving relevant mitigation techniques" query in
# sparql_toolQ.md, but parameterized by {concern_iri} instead of the
# hardcoded core:BiasPerpetuation example. Uses the SWRL-derived
# `mitigatedWith` relation, so it only returns results once the ontology's
# reasoning pass has been run to materialize that relation.
MITIGATION_TECHNIQUE_FOR_CONCERN_QUERY = """
PREFIX rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#>
PREFIX owl: <http://www.w3.org/2002/07/owl#>
PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
PREFIX xsd: <http://www.w3.org/2001/XMLSchema#>
PREFIX core: <https://purl.org/fairops/core#>
PREFIX skos: <http://www.w3.org/2004/02/skos/core#>
PREFIX indiv: <https://purl.org/fairops/indiv#>

SELECT DISTINCT ?mitTech
WHERE {{
    ?mitTech core:enforces <{metric_iri}> .
}}
"""

@st.cache_resource
def _load_ontology_graph():
    graph = Graph()
    graph.parse(FAIROPS_ONTOLOGY_PATH, format="turtle")
    graph.parse(FAIROPS_INDIVIDUALS_PATH, format="turtle")
    return graph


def run_sparql_query(query):
    """Execute an arbitrary SPARQL SELECT query against the fairops ontology
    graph and return its results as a list of {var_name: value} dicts, one
    per row, with values stringified (None for unbound variables)."""
    graph = _load_ontology_graph()
    result = graph.query(query)
    return [
        {str(var): (str(row[var]) if row[var] is not None else None) for var in result.vars}
        for row in result
    ]


# Competency questions from the fairops ontology docs
# (fairness_ontology/fairops/SPARQL queries/sparql_CQ.md), illustrated on the
# example scenario described there: an AI-enabled hiring recommendation
# system (Application Domain: Human Resources, AI Type of Use:
# Recommendation, Fairness Concern: PopularItemsOverrecommended, Fairness
# Notion: StatisticalParity).
COMPETENCY_QUESTIONS = [
    {
        "id": "Q1",
        "text": "Which legal requirements are applicable to the considered AI context?",
        # {application_domain_iri} is bound at render time to the IRI
        # currently selected in the "Application Domain" combobox (see
        # "params" below and render_competency_questions()), so the query
        # tracks whichever domain is picked instead of a hardcoded example.
        "query": """PREFIX rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#>
PREFIX owl: <http://www.w3.org/2002/07/owl#>
PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
PREFIX xsd: <http://www.w3.org/2001/XMLSchema#>
PREFIX core: <https://purl.org/fairops/core#>
PREFIX skos: <http://www.w3.org/2004/02/skos/core#>
SELECT ?legalRequirement ?com
WHERE {{
    <{application_domain_iri}> core:triggers ?legalRequirement .
    ?legalRequirement rdfs:comment ?com
}}""",
        # Maps each {placeholder} in "query" to the st.session_state key it
        # should be formatted with.
        "params": {"application_domain_iri": "application_domain"},
    },
    {
        "id": "Q2",
        "text": "Which fairness concerns are associated with the given AI type of use?",
        "query": """PREFIX rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#>
PREFIX owl: <http://www.w3.org/2002/07/owl#>
PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
PREFIX xsd: <http://www.w3.org/2001/XMLSchema#>
PREFIX core: <https://purl.org/fairops/core#>
PREFIX skos: <http://www.w3.org/2004/02/skos/core#>
SELECT  DISTINCT ?fc ?def
WHERE {
    ?fc rdf:type ?type .
    ?type rdfs:subClassOf* core:FairnessConcern .

    ?fc core:arisesIn core:Recommendation ;
        skos:definition ?def .
}""",
    },
    {
        "id": "Q3",
        "text": "Which fairness notions address a specific concern?",
        "query": """PREFIX rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#>
PREFIX owl: <http://www.w3.org/2002/07/owl#>
PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
PREFIX xsd: <http://www.w3.org/2001/XMLSchema#>
PREFIX core: <https://purl.org/fairops/core#>
PREFIX skos: <http://www.w3.org/2004/02/skos/core#>
SELECT ?notion ?def
WHERE {
    core:PopularItemsOverrecommended core:isAddressedWith ?notion .
    ?notion core:scientificArtifactDescription ?def.
}""",
    },
    {
        "id": "Q4",
        "text": "Which fairness notions conflict with a selected one?",
        "query": """PREFIX rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#>
PREFIX owl: <http://www.w3.org/2002/07/owl#>
PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
PREFIX xsd: <http://www.w3.org/2001/XMLSchema#>
PREFIX core: <https://purl.org/fairops/core#>
PREFIX skos: <http://www.w3.org/2004/02/skos/core#>
PREFIX indiv: <https://purl.org/fairops/indiv#>
SELECT ?conflNotion
WHERE {
    indiv:StatisticalParity core:conflictsWith ?conflNotion .
}""",
    },
    {
        "id": "Q5",
        "text": "Which fairness metrics are appropriate to a specific concern in the given AI context?",
        "query": """PREFIX rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#>
PREFIX owl: <http://www.w3.org/2002/07/owl#>
PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
PREFIX xsd: <http://www.w3.org/2001/XMLSchema#>
PREFIX core: <https://purl.org/fairops/core#>
PREFIX skos: <http://www.w3.org/2004/02/skos/core#>
SELECT ?metric
WHERE {
    core:PopularItemsOverrecommended core:isQuantifiedBy ?metric .
}""",
    },
    {
        "id": "Q6",
        "text": "Which mitigation techniques addressing a concern are operationally feasible under the available deployment constraints (no feasible retraining or fine-tuning action)?",
        "query": """PREFIX rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#>
PREFIX owl: <http://www.w3.org/2002/07/owl#>
PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
PREFIX xsd: <http://www.w3.org/2001/XMLSchema#>
PREFIX core: <https://purl.org/fairops/core#>
PREFIX skos: <http://www.w3.org/2004/02/skos/core#>
SELECT ?mitTech ?def
WHERE {
    core:PopularItemsOverrecommended core:mitigatedWith ?mitTech .
    ?mitTech core:scientificArtifactDescription ?def ;
            rdf:type ?type .
    ?type rdfs:subClassOf* ?rootMitTech .

    FILTER(
        ?rootMitTech IN (
            core:GreyBoxScores,
            core:BlackBoxDecisionOnly,
            core:HumanOversightMitigation
        )
    )
}""",
    },
    {
        "id": "Q7",
        "text": "Which evidence artifacts are required to support compliance and auditing?",
        "query": """PREFIX rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#>
PREFIX owl: <http://www.w3.org/2002/07/owl#>
PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
PREFIX xsd: <http://www.w3.org/2001/XMLSchema#>
PREFIX core: <https://purl.org/fairops/core#>
PREFIX skos: <http://www.w3.org/2004/02/skos/core#>
SELECT ?evidenceArtifact ?def
WHERE {
    core:PopularItemsOverrecommended core:requires ?evidenceArtifact .
    ?evidenceArtifact skos:definition ?def.
}""",
    },
]


def render_competency_questions():
    """One row per fairops competency question: the question text plus a
    button that runs its SPARQL query against the ontology graph and shows
    the results underneath the row (kept in session_state so results persist
    across reruns triggered by other widgets).

    A question's "params" dict (placeholder -> session_state key) is
    resolved and formatted into its "query" template at run time, so e.g. Q1
    tracks whichever IRI is currently selected in the "Application Domain"
    combobox. The button is disabled while a required selection is missing.
    """
    for cq in COMPETENCY_QUESTIONS:
        params = cq.get("params", {})
        missing = [
            session_key
            for session_key in params.values()
            if not st.session_state.get(session_key)
        ]

        question_col, button_col = st.columns([9, 2], vertical_alignment="center")
        with question_col:
            st.markdown(f"**{cq['id']}.** {cq['text']}")
        with button_col:
            run_clicked = st.button(
                "▶ Run query", key=f"run_cq_{cq['id']}", disabled=bool(missing)
            )

        results_key = f"cq_results_{cq['id']}"
        if missing:
            st.caption(f"Select a value for {', '.join(missing)} above to run this query.")
        elif run_clicked:
            format_kwargs = {
                placeholder: st.session_state[session_key]
                for placeholder, session_key in params.items()
            }
            query = cq["query"].format(**format_kwargs) if params else cq["query"]
            try:
                st.session_state[results_key] = run_sparql_query(query)
            except Exception as exc:
                st.session_state[results_key] = exc

        results = st.session_state.get(results_key)
        if isinstance(results, Exception):
            st.error(f"Query failed: {results}")
        elif results is not None:
            if results:
                st.dataframe(pd.DataFrame(results), use_container_width=True)
            else:
                st.info("Query returned no results.")


def _humanize(local_name):
    # Keep acronyms together (ICTSecurity -> ICT Security) before splitting
    # the remaining lower-to-upper word boundaries (RetailAndECommerce ->
    # Retail And E Commerce).
    spaced = re.sub(r"([A-Z]+)([A-Z][a-z])", r"\1 \2", local_name)
    spaced = re.sub(r"([a-z0-9])([A-Z])", r"\1 \2", spaced)
    return spaced


def _local_name(iri):
    return str(iri).split("#")[-1]


def label_for_iri(iri):
    return _humanize(_local_name(iri))


def _query_individual_iris(query):
    graph = _load_ontology_graph()
    iris = {str(row.individual) for row in graph.query(query)}
    return sorted(iris, key=label_for_iri)


def get_application_domains():
    return _query_individual_iris(APPLICATION_DOMAIN_QUERY)


def get_ai_tasks():
    return _query_individual_iris(AI_TASK_QUERY)


def get_ai_type_of_use():
    return _query_individual_iris(AI_TYPE_OF_USE_QUERY)


def get_fairness_concerns(ai_type_of_use_iri):
    graph = _load_ontology_graph()
    query = FAIRNESS_CONCERN_QUERY.format(ai_type_of_use_iri=ai_type_of_use_iri)
    concerns = {}
    for row in graph.query(query):
        iri = str(row.fc)
        definition = str(row["def"]) if row["def"] is not None else None
        existing = concerns.get(iri)
        if existing is None or (definition and not existing["definition"]):
            concerns[iri] = {
                "iri": iri,
                "label": label_for_iri(iri),
                "definition": definition,
            }
    return sorted(concerns.values(), key=lambda c: c["label"])


def get_fairness_notions(concern_iri):
    graph = _load_ontology_graph()
    query = FAIRNESS_NOTION_QUERY.format(concern_iri=concern_iri)
    iris = {str(row.notion) for row in graph.query(query)}
    notions = [{"iri": iri, "label": label_for_iri(iri)} for iri in iris]
    return sorted(notions, key=lambda n: n["label"])


def get_fairness_metrics(notion_iri):
    graph = _load_ontology_graph()
    query = FAIRNESS_METRIC_QUERY.format(notion_iri=notion_iri)
    iris = {str(row.fm) for row in graph.query(query)}
    metrics = [{"iri": iri, "label": label_for_iri(iri)} for iri in iris]
    return sorted(metrics, key=lambda m: m["label"])


def get_mitigation_techniques_for_concern(concern_iri):
    graph = _load_ontology_graph()
    query = MITIGATION_TECHNIQUE_FOR_CONCERN_QUERY.format(metric_iri=concern_iri)
    iris = {str(row.mitTech) for row in graph.query(query)}
    techniques = [{"iri": iri, "label": label_for_iri(iri)} for iri in iris]
    if len(techniques) == 0:
        techniques = [{"iri": "N/A", "label": "Mitigation sample"}]
    return sorted(techniques, key=lambda t: t["label"])


def session_state_params(current_product, current_framework):
    if "current_product" not in st.session_state:
        st.session_state.current_product = current_product
    if "current_framework" not in st.session_state:
        st.session_state.current_framework = current_framework


def get_pipeline_operations():
    with open(pipeline_definitions_folder, "r") as yaml_file:
        pipeline_configs = yaml.safe_load(yaml_file)
    pipeline_ai_operations = pipeline_configs["ai_operations"]
    pipeline_data_operations = [
        elem["operations"]
        for elem in pipeline_ai_operations
        if elem["stage"] == "data_preparation"
    ]
    pipeline_model_operations = [
        elem["operations"]
        for elem in pipeline_ai_operations
        if elem["stage"] == "modelling"
    ]
    pipeline_operationalisation_ops = [
        elem["operations"]
        for elem in pipeline_ai_operations
        if elem["stage"] == "operationalization"
    ]
    all_pipeline_operations = (
        pipeline_data_operations[0]
        + pipeline_model_operations[0]
        + pipeline_operationalisation_ops[0]
    )
    return all_pipeline_operations


CASCADE_COLORS = ["#dbeafe", "#bfdbfe", "#93c5fd"]


def render_cascade_checkbox(label, key, level=0, help=None):
    """One row of an indented, parent-reveals-children checkbox tree.

    Deeper levels get more left indent and a darker card color from
    CASCADE_COLORS, matching the checkbox+card visual language already used
    by populate_stages().
    """
    color = CASCADE_COLORS[min(level, len(CASCADE_COLORS) - 1)]
    indent = level * 32
    checkbox_col, card_col, _, _ = st.columns([1, 5, 5, 5], gap="small")
    with checkbox_col:
        st.markdown(
            f"<div style='height:14px; margin-left:{indent}px'></div>",
            unsafe_allow_html=True,
        )
        checked = st.checkbox(
            label, key=key, help=help, value=False, label_visibility="collapsed"
        )
    with card_col:
        st.markdown(
            f"""
            <div style="
                margin-left:{indent}px;
                min-height:40px;
                background-color:{color};
                border-radius:10px;
                padding:8px 12px;
                display:flex;
                align-items:center;
            ">{label}</div>
            """,
            unsafe_allow_html=True,
        )
    return checked


def render_cascade_question(text, options, key, level=0):
    """One row of a hierarchical Q&A flow: a question card whose radio-button
    answer determines which follow-up question(s) appear next. Mirrors the
    indent/color scheme of render_cascade_checkbox. index=None means no
    alternative is pre-selected, so children only appear once the user
    actively answers.
    """
    color = CASCADE_COLORS[min(level, len(CASCADE_COLORS) - 1)]
    indent = level * 32
    question_col, answer_col = st.columns([6, 6], gap="small")
    with question_col:
        st.markdown(
            f"""
            <div style="
                margin-left:{indent}px;
                min-height:40px;
                background-color:{color};
                border-radius:10px;
                padding:8px 12px;
                display:flex;
                align-items:center;
            ">{text}</div>
            """,
            unsafe_allow_html=True,
        )
    with answer_col:
        answer = st.radio(
            text,
            options,
            key=key,
            index=None,
            horizontal=True,
            label_visibility="collapsed",
        )
    return answer


# Stage -> level/group/colors, matching the pre/in/post-processing
# mitigation groups (see MITIGATION_GROUP_STYLES in
# pages/2_new_aiproduct.py): yellow for data_preparation (Pre-processing,
# level 1), green for modelling (In-processing, level 2), blue for
# operationalization (Post-processing, level 3). `level` orders the stages
# so populate_stages() can tell which ones sit "before" a recommended
# mitigation group.
STAGE_MITIGATION_STYLES = {
    "data_preparation": {
        "level": 1,
        "group": "Pre-processing",
        "bg": "#fef9c3",
        "border": "#eab308",
    },
    "modelling": {
        "level": 2,
        "group": "In-processing",
        "bg": "#dcfce7",
        "border": "#10b981",
    },
    "operationalization": {
        "level": 3,
        "group": "Post-processing",
        "bg": "#dbeafe",
        "border": "#3b82f6",
    },
}

# Muted style applied to stages that come before the resource-aware flow's
# recommended mitigation group -- their operations are shown read-only and
# forced unselected rather than colored per STAGE_MITIGATION_STYLES.
DISABLED_STAGE_STYLE = {"bg": "#f3f4f6", "border": "#d1d5db"}

_DEFAULT_STAGE_META = {"level": 0, "group": None, "bg": "#dbeafe", "border": "#3b82f6"}

def import_from_path(module_name, file_path):
    """Import a module given its name and file path."""
    spec = importlib.util.spec_from_file_location(module_name, file_path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def load_method_content(
    method_name,
    current_product,
    current_framework="local",
    step_operations_module="data_preparation.py",
):
    product_operations_file = os.path.join(
        USE_CASES_FOLDER,
        current_product,
        "src",
        f"{current_framework}_platform",
        step_operations_module,
    )
    curr_module = import_from_path("curr_module", product_operations_file)
    func = getattr(curr_module, method_name)
    source_text = inspect.getsource(func)
    return source_text

@st.cache_data(show_spinner=False)
def load_aipc_config(current_product, current_framework):
    """Full parsed aipc_<framework>.yaml for a use case (operations +
    artifacts). Empty dict if the product hasn't been configured for that
    framework yet."""
    config_path = os.path.join(
        USE_CASES_FOLDER, current_product, "metadata", f"aipc_{current_framework}.yaml"
    )
    if not os.path.exists(config_path):
        return {}
    with open(config_path, "r") as yaml_file:
        return yaml.safe_load(yaml_file) or {}


def get_operation_implementations(op_type, current_product, current_framework):
    """aipc_*.yaml operation entries wired to a pipeline operation `type`
    (e.g. "model_training") -- there can be several per type (baseline vs.
    fairness-aware variants), see aipc_local.yaml."""
    aipc_config = load_aipc_config(current_product, current_framework)
    return [
        entry
        for entry in aipc_config.get("operations", [])
        if entry.get("type") == op_type
    ]


def populate_stages(
    pipeline_configs,
    createview=False,
    recommended_group=None,
    current_product="recruitment",
    current_framework="local",
):
    recommended_level = next(
        (
            meta["level"]
            for meta in STAGE_MITIGATION_STYLES.values()
            if meta["group"] == recommended_group
        ),
        None,
    )
    cols = st.columns(3)
    for i, stage in enumerate(pipeline_configs["ai_operations"]):
        stage_meta = STAGE_MITIGATION_STYLES.get(stage["stage"], _DEFAULT_STAGE_META)
        is_read_only = (
            recommended_level is not None and stage_meta["level"] < recommended_level
        )
        stage_style = DISABLED_STAGE_STYLE if is_read_only else stage_meta
        with cols[i]:
            for j, operation in enumerate(stage["operations"]):
                op_type = list(operation.keys())[0]
                articles = (operation[op_type] or {}).get("article") or []
                label = op_type.upper().replace("_", " ")

                if createview:
                    _, card_col = st.columns([1, 9], gap="small")
                else:
                    card_col = st.container()
                if is_read_only:
                    label = f"🔒 {label}"
                with card_col:
                    # Every operation is a real st.button styled to match the
                    # colored rectangle (background/border/badges via the
                    # `st-key-<key>` CSS hook + an absolute-positioned badge
                    # overlay), so it triggers an in-place rerun on click.
                    # A plain markdown div can't do this -- onclick doesn't
                    # work here (Streamlit sanitizes unsafe_allow_html via
                    # DOMPurify, which strips inline event-handler
                    # attributes), so a real widget is the only way to get a
                    # working click.
                    card_key = f"op_card_{op_type}"
                    btn_key = f"op_btn_{op_type}"
                    badge_key = f"op_badge_{op_type}"
                    st.markdown(
                        f"""
                        <style>
                        .st-key-{card_key} {{
                            position: relative;
                            /* The badge lives in its own sub-block (needed
                            for the position:relative escape trick below),
                            which still eats the default 16px inter-element
                            gap even though it renders nothing in normal
                            flow -- zero it out. */
                            gap: 0;
                            {"margin-top: -8px;" if j > 0 else ""}
                        }}
                        .st-key-{btn_key} button {{
                            background-color: {stage_style['bg']};
                            border: 1px solid {stage_style['border']};
                            opacity: {0.55 if is_read_only else 1};
                            justify-content: flex-start;
                            {"padding-top: 20px;" if articles else ""}
                        }}
                        .st-key-{btn_key} button p {{
                            text-align: left;
                            font-weight: 700;
                        }}
                        .st-key-{btn_key} button:hover:not(:disabled) {{
                            border-color: {stage_style['border']};
                            color: {stage_style['border']};
                        }}
                        /* Streamlit gives every element's own wrapper
                        (.stElementContainer) position:relative by default,
                        which -- being nearer than .st-key-{card_key} above --
                        wins as the containing block for the badge's
                        position:absolute div below, anchoring it to this
                        badge's own (zero-height) flow slot right after the
                        button instead of to the button itself. Neutralizing
                        position here on both the keyed wrapper and its inner
                        element-container lets the badge escape up to
                        .st-key-{card_key}, whose box actually matches the
                        button's. */
                        .st-key-{badge_key}, .st-key-{badge_key} .stElementContainer {{
                            position: static;
                        }}
                        </style>
                        """,
                        unsafe_allow_html=True,
                    )
                    card = st.container(key=card_key)
                    with card:
                        clicked = st.button(
                            label,
                            key=btn_key,
                            use_container_width=True,
                            disabled=is_read_only,
                        )
                        if articles:
                            # No blank lines between concatenated spans --
                            # Streamlit's markdown renderer treats a blank
                            # line inside unsafe HTML as a paragraph break
                            # and wraps the next badge in a stray <p>,
                            # breaking it out of the flex row (and out of
                            # vertical alignment with the others).
                            circles = "".join(
                                f'<span style="width:20px; height:20px; '
                                "border-radius:50%; background-color:#6b7280; "
                                "color:#ffffff; display:flex; align-items:center; "
                                "justify-content:center; font-size:0.65em; "
                                'font-weight:600; flex-shrink:0;">'
                                f"{art}</span>"
                                for art in articles
                            )
                            badge_container = st.container(key=badge_key)
                            with badge_container:
                                st.markdown(
                                    f"""
                                    <div style="
                                        position:absolute;
                                        top:6px;
                                        right:10px;
                                        display:flex;
                                        align-items:center;
                                        gap:4px;
                                        opacity:{0.55 if is_read_only else 1};
                                        pointer-events:none;
                                    "><span style="font-size:0.7em; color:#374151;">Art.</span>{circles}</div>
                                    """,
                                    unsafe_allow_html=True,
                                )

                    if clicked:
                        # Loads the operation category's wired
                        # implementation(s); show_operation_implementation()
                        # (pages/2_new_aiproduct.py) renders them from here.
                        st.session_state["selected_operation_implementation"] = {
                            "op_type": op_type,
                            "current_product": current_product,
                            "current_framework": current_framework,
                            "entries": get_operation_implementations(
                                op_type, current_product, current_framework
                            ),
                        }

def get_function_source(file_path, function_name):
    with open(file_path, "r") as f:
        source = f.read()
    tree = ast.parse(source)
    for node in ast.iter_child_nodes(tree):
        if isinstance(node, ast.FunctionDef) and node.name == function_name:
            start_line = node.lineno - 1
            end_line = node.end_lineno
            lines = source.splitlines()
            return "\n".join(lines[start_line:end_line])
    return None