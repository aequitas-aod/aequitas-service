# streamlit: page_name = "Dashboard"
import os
import yaml
import pandas as pd
import streamlit as st
from utils import get_pipeline_operations, session_state_params, populate_stages

current_folder = os.path.dirname(os.path.abspath(__file__))
parent_folder = os.path.dirname(current_folder)
parent_folder = os.path.dirname(parent_folder)
pipeline_definitions_folder = os.path.join(
    parent_folder, "framework/temlops/config/pipeline_definitions.yaml"
)
USE_CASES_FOLDER = os.path.join(parent_folder, "framework/temlops/use_cases")
use_cases_list = os.listdir(USE_CASES_FOLDER)
platform_list = ["local", "dh"]


def show_dashboard(current_product, current_framework):
    st.title(
        "Dashboard: AI Product Operations by Requirements Dimensions",
        anchor="dashboard",
    )
    current_product = st.sidebar.selectbox("AI Products list", use_cases_list)
    current_framework = st.sidebar.selectbox(
        "Tool governance platform", platform_list, index=0
    )
    session_state_params(current_product, current_framework)

    product_config_file = os.path.join(
        USE_CASES_FOLDER, current_product, "metadata", f"aipc_{current_framework}.yaml"
    )
    with open(product_config_file, "r") as yaml_file:
        aipc_configs = yaml.safe_load(yaml_file)

    with open(pipeline_definitions_folder, "r") as yaml_file:
        pipeline_configs = yaml.safe_load(yaml_file)

    requirements_dimensions = aipc_configs["requirements_dimensions"]
    populate_stages(pipeline_configs)
    for requirement in requirements_dimensions:
        status = read_operations_status(pipeline_configs, aipc_configs, requirement)
        show_operations_status(pipeline_configs, requirement, status)


def show_operations_status(pipeline_configs, requirement, status):
    with st.expander(f"{requirement.upper()} OPERATIONS", expanded=True):
        cols_req = st.columns(3)
        for i, stage in enumerate(pipeline_configs["ai_operations"]):
            with cols_req[i]:
                for j, operation in enumerate(stage["operations"]):
                    op_type = list(operation.keys())[0]
                    declared_operations = list(
                        filter(lambda x: x["operation_type"] == op_type, status)
                    )[0]["passed_operations"]
                    passed = len(declared_operations) > 0
                    not_active = list(
                        filter(lambda x: x["operation_type"] == op_type, status)
                    )[0]["not_active"]

                    st.markdown(
                        f"""
                            <div style="
                                margin-left:{j*40}px;
                                height:60px;
                                overflow-y:auto;
                                width:{450 - j*40}px;
                                background-color:{'#e6f4ea' if passed else '#d3d3d3' if not_active else '#FFE5E5'};
                                border-radius:10px;
                                padding:10px;
                            ">
                                {';</br> '.join(declared_operations) if passed  else '🚫' if not_active else "✗"}
                            </div>
                            """,
                        unsafe_allow_html=True,
                    )


def read_operations_status(pipeline_configs, aipc_configs, requirement):
    status = []
    not_active_ops = aipc_configs["excluded_operations"]
    for i, stage in enumerate(pipeline_configs["ai_operations"]):
        for j, operation in enumerate(stage["operations"]):
            op_type = list(operation.keys())[0]
            passed_operations = [
                op["id"]
                for op in aipc_configs["operations"]
                if op["requirement_dimension"] == requirement and op["type"] == op_type
            ]
            status.append(
                {
                    "operation_type": op_type,
                    "passed_operations": passed_operations,
                    "not_active": op_type in not_active_ops,
                }
            )
    return status


if __name__ == "__main__":
    current_product = 0
    current_framework = 0
    if "current_product" in st.session_state:
        current_product = st.session_state.current_product
    if "current_framework" in st.session_state:
        current_framework = st.session_state.current_framework
    st.set_page_config(page_title="Dashboard", layout="wide")
    show_dashboard(current_product, current_framework)
