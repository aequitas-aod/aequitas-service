import os
import re
from dotenv import load_dotenv, find_dotenv
import yaml
import streamlit as st
from rdflib import Graph

load_dotenv(find_dotenv())

current_folder = os.path.dirname(os.path.abspath(__file__))
parent_folder = os.path.dirname(current_folder)
pipeline_definitions_folder = os.path.join(
    parent_folder, "framework/temlops/config/pipeline_definitions.yaml"
)

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


def populate_stages(pipeline_configs, createview=False, recommended_group=None):
    recommended_level = next(
        (
            meta["level"]
            for meta in STAGE_MITIGATION_STYLES.values()
            if meta["group"] == recommended_group
        ),
        None,
    )
    cols = st.columns(3)
    selected_operations = {}
    for i, stage in enumerate(pipeline_configs["ai_operations"]):
        stage_meta = STAGE_MITIGATION_STYLES.get(stage["stage"], _DEFAULT_STAGE_META)
        is_read_only = (
            recommended_level is not None and stage_meta["level"] < recommended_level
        )
        stage_style = DISABLED_STAGE_STYLE if is_read_only else stage_meta
        with cols[i]:
            for j, operation in enumerate(stage["operations"]):
                op_type = list(operation.keys())[0]
                article = "10.2"
                card_width = 450 - j * 40

                if createview:
                    checkbox_col, card_col = st.columns([1, 9], gap="small")
                    with checkbox_col:
                        st.markdown(
                            "<div style='height:14px'></div>", unsafe_allow_html=True
                        )
                        checkbox_key = f"op_select_{op_type}"
                        if is_read_only:
                            # Force the persisted widget value back to
                            # unselected -- setting `value=` alone only seeds
                            # a *new* key, it won't override state from a
                            # previous, non-restricted run.
                            st.session_state[checkbox_key] = False
                        checked = st.checkbox(
                            op_type,
                            value=not is_read_only,
                            key=checkbox_key,
                            label_visibility="collapsed",
                            disabled=is_read_only,
                        )
                    selected_operations[op_type] = checked
                    card_width -= 60
                else:
                    card_col = st.container()

                with card_col:
                    label = op_type.upper().replace("_", " ")
                    if is_read_only:
                        label = f"🔒 {label}"
                    st.markdown(
                        f"""
                        <div style="
                            margin-left:{j*40}px;
                            height:50px;
                            overflow-y:auto;
                            width:{card_width}px;
                            background-color:{stage_style['bg']};
                            border-left:4px solid {stage_style['border']};
                            border-radius:10px;
                            padding:10px;
                            display:flex;
                            align-items:center;
                            gap:10px;
                            opacity:{0.55 if is_read_only else 1};
                        ">
                            {label}
                            </div>
                        """,
                        unsafe_allow_html=True,
                    )
                            # <span style="
                            #     display:inline-flex;
                            #     align-items:center;
                            #     justify-content:center;
                            #     min-width:36px;
                            #     height:36px;
                            #     border-radius:50%;
                            #     background-color:#60a5fa;
                            #     color:#ffffff;
                            #     font-size:12px;
                            #     font-weight:700;
                            #     padding:2px 4px;
                            #     flex-shrink:0;
                            #     margin-left:auto;
                            # ">{article}</span>                                             
    return selected_operations
