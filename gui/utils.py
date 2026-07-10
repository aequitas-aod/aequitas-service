import os
import re
import yaml
import streamlit as st
from rdflib import Graph

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
    "/home/albana/Desktop/Albana/DataScience/UniBO/AIFactory/Projects/fairness_ontology/fairops/docs/fairops.ttl",
)

APPLICATION_DOMAIN_QUERY = """
PREFIX rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#>
PREFIX core: <https://purl.org/fairops/core#>

SELECT ?individual
WHERE {
    ?individual rdf:type core:ApplicationDomain .
}
"""


@st.cache_resource
def _load_ontology_graph():
    graph = Graph()
    graph.parse(FAIROPS_ONTOLOGY_PATH, format="turtle")
    return graph


def _humanize(local_name):
    # Keep acronyms together (ICTSecurity -> ICT Security) before splitting
    # the remaining lower-to-upper word boundaries (RetailAndECommerce ->
    # Retail And E Commerce).
    spaced = re.sub(r"([A-Z]+)([A-Z][a-z])", r"\1 \2", local_name)
    spaced = re.sub(r"([a-z0-9])([A-Z])", r"\1 \2", spaced)
    return spaced


def get_application_domains():
    graph = _load_ontology_graph()
    local_names = [
        str(row.individual).split("#")[-1] for row in graph.query(APPLICATION_DOMAIN_QUERY)
    ]
    return sorted(_humanize(name) for name in local_names)


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


def populate_stages(pipeline_configs, createview=False):
    colors = ["#dbeafe", "#bfdbfe", "#93c5fd"]
    cols = st.columns(3)
    selected_operations = {}
    for i, stage in enumerate(pipeline_configs["ai_operations"]):
        with cols[i]:
            for j, operation in enumerate(stage["operations"]):
                op_type = list(operation.keys())[0]
                article = "10.2"
                card_width = 450 - j * 40

                if createview:
                    checkbox_col, card_col = st.columns([1, 9], gap="small")
                    with checkbox_col:
                        st.markdown("<div style='height:14px'></div>", unsafe_allow_html=True)
                        checked = st.checkbox(
                            op_type,
                            value=True,
                            key=f"op_select_{op_type}",
                            label_visibility="collapsed",
                        )
                    selected_operations[op_type] = checked
                    card_width -= 60
                else:
                    card_col = st.container()

                with card_col:
                    st.markdown(
                        f"""
                        <div style="
                            margin-left:{j*40}px;
                            height:50px;
                            overflow-y:auto;
                            width:{card_width}px;
                            background-color:#dbeafe;
                            border-radius:10px;
                            padding:10px;
                            display:flex;
                            align-items:center;
                            gap:10px;
                        ">
                            {op_type.upper().replace("_", " ")}
                            <span style="
                                display:inline-flex;
                                align-items:center;
                                justify-content:center;
                                min-width:36px;
                                height:36px;
                                border-radius:50%;
                                background-color:#60a5fa;
                                color:#ffffff;
                                font-size:12px;
                                font-weight:700;
                                padding:2px 4px;
                                flex-shrink:0;
                                margin-left:auto;
                            ">{article}</span>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )
    return selected_operations
