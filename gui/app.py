# streamlit: page_name = "Main"
import os
import re
import ast
import sys
import yaml
import json
import inspect
import pandas as pd
import streamlit as st
import importlib.util
from pathlib import Path
from utils import (
    get_pipeline_operations,
    session_state_params,
    load_method_content,
    import_from_path,
    get_function_source
)
from streamlit_ace import st_ace
from streamlit_option_menu import option_menu
from temlops.src.artifact_types import Data, Configuration, Report, Model


current_folder = os.path.dirname(os.path.abspath(__file__))
parent_folder = os.path.dirname(current_folder)
pipeline_definitions_folder = os.path.join(
    parent_folder, "framework/temlops/config/pipeline_definitions.yaml"
)
USE_CASES_FOLDER = os.path.join(parent_folder, "framework/temlops/use_cases")
TOOLS_CATALOG_FOLDER = os.path.join(parent_folder, "tools_catalog")
use_cases_list = os.listdir(USE_CASES_FOLDER)
platform_list = ["local", "dh"]

sys.path.append(USE_CASES_FOLDER)


def stages_settings(stage):
    if stage == "Data preparation":
        st.title("📊 Stage 1: Data preparation")
        selected_operation = option_menu(
            "Operations",
            [
                "Data profiling",
                "Data validation",
                "Data preprocessing",
                "Data documentation",
            ],
            menu_icon="sliders",
            default_index=0,
            orientation="horizontal",
        )
        selected_operation = selected_operation.replace(" ", "_").lower()
        step_operations_module = "data_preparation.py"
        return selected_operation, step_operations_module
    elif stage == "Modeling":
        st.title("🛠️ Stage 2: Modeling")
        selected_operation = option_menu(
            "Operations",
            [
                "Feature engineering",
                "Model training",
                "Model evaluation",
                "Model validation",
                "Model documentation",
            ],
            menu_icon="sliders",
            default_index=0,
            orientation="horizontal",
        )
        selected_operation = selected_operation.replace(" ", "_").lower()
        step_operations_module = "modelling.py"
        return selected_operation, step_operations_module
    elif stage == "Operationalisation":
        st.title("⚙️ Stage 3: Operationalization")
        selected_operation = option_menu(
            "Operations",
            [
                "Model deployment",
                "Model monitoring",
                "Production data monitoring",
                "System monitoring",
                "Pre inference transformations",
                "Post inference transformations",
            ],
            menu_icon="sliders",
            default_index=0,
            orientation="horizontal",
        )
        selected_operation = selected_operation.replace(" ", "_").lower()
        step_operations_module = "operationalization.py"
        return selected_operation, step_operations_module


def show_stage(
    current_step: str = "Data preparation",
    selected_aspect: str = "Baseline",
    current_product: str = "tabular",
    current_framework: str = "local",
):
    selected_operation, step_operations_module = stages_settings(current_step)
    product_config_file = os.path.join(
        USE_CASES_FOLDER, current_product, "metadata", f"aipc_{current_framework}.yaml"
    )
    if not os.path.exists(product_config_file):
        st.error(
            f"Configuration file for {current_product} does not exist for {current_framework} platform. Please, configure the AI product properly",
            icon="🚨",
        )
        return
    with open(product_config_file, "r") as yaml_file:
        aipc_configs = yaml.safe_load(yaml_file)
        # operations
        operations = aipc_configs["operations"]
        data_operations = [
            elem
            for elem in operations
            if elem["stage"] == "data_preparation"
            and elem["requirement_dimension"] == selected_aspect.lower()
        ]
        model_operations = [
            elem
            for elem in operations
            if elem["stage"] == "modelling"
            and elem["requirement_dimension"] == selected_aspect.lower()
        ]
        operationalisation_ops = [
            elem
            for elem in operations
            if elem["stage"] == "operationalization"
            and elem["requirement_dimension"] == selected_aspect.lower()
        ]
        # artifacts
        artifacts = aipc_configs["artifacts"]
        data_artifacts = artifacts["data"]
        model_artifacts = artifacts["model"]
        report_artifacts = artifacts["report"]
        configuration_artifacts = artifacts["configuration"]

        if current_step == "Data preparation":
            populate_frames(
                data_operations,
                selected_operation,
                step_operations_module,
                current_product,
                current_framework,
                model_artifacts,
                data_artifacts,
                configuration_artifacts,
                report_artifacts,
            )
            # visualize data artifacts
            # populate_data_artifacts(data_artifacts)
        elif current_step == "Modeling":
            populate_frames(
                model_operations,
                selected_operation,
                step_operations_module,
                current_product,
                current_framework,
                model_artifacts,
                data_artifacts,
                configuration_artifacts,
                report_artifacts,
            )
        elif current_step == "Operationalisation":
            populate_frames(
                operationalisation_ops,
                selected_operation,
                step_operations_module,
                current_product,
                current_framework,
                model_artifacts,
                data_artifacts,
                configuration_artifacts,
                report_artifacts,
            )


def show_artifacts(
    current_step: str = "Data preparation",
    selected_aspect: str = "Baseline",
    current_product: str = "tabular",
    current_framework: str = "local",
):
    product_config_file = os.path.join(
        USE_CASES_FOLDER, current_product, "metadata", f"aipc_{current_framework}.yaml"
    )
    if not os.path.exists(product_config_file):
        st.error(
            f"Configuration file for {current_product} does not exist for {current_framework} platform. Please, configure the AI product properly",
            icon="🚨",
        )
        return
    with open(product_config_file, "r") as yaml_file:
        aipc_configs = yaml.safe_load(yaml_file)
        # artifacts
        artifacts = aipc_configs["artifacts"]
        data_artifacts = artifacts["data"]
        model_artifacts = artifacts["model"]
        report_artifacts = artifacts["report"]
        configuration_artifacts = artifacts["configuration"]
        # visualize data artifacts
        with st.expander(f"Data Artifacts", expanded=True):
            populate_data_artifacts(data_artifacts)
        with st.expander(f"Model Artifacts", expanded=False):
            populate_artifacts(model_artifacts)
        with st.expander(f"Report Artifacts", expanded=False):
            populate_artifacts(report_artifacts)
        with st.expander(f"Configuration Artifacts", expanded=False):
            populate_artifacts(configuration_artifacts)


def extract_section_by_title(md_text, title):
    pattern = rf"##\s+{re.escape(title)}\n(.*?)(?=\n##\s|\Z)"
    return re.search(pattern, md_text, re.S).group(1).strip()


def populate_frames(
    operations,
    selected_data_operation,
    step_operations_module,
    current_product,
    current_framework,
    model_artifacts,
    data_artifacts,
    configuration_artifacts,
    report_artifacts,
):
    with open(pipeline_definitions_folder, "r") as yaml_file:
        pipeline_configs = yaml.safe_load(yaml_file)
        all_pipeline_operations = get_pipeline_operations()
        selected_op = [
            elem
            for elem in all_pipeline_operations
            if list(elem.keys())[0] == selected_data_operation
        ]
        desc = selected_op[0][selected_data_operation]["desc"]
        st.markdown(
            f"""
            <div style="
                border: 2px solid #d1d5db;
                border-radius: 10px;
                padding: 16px;
                background-color: #f3f4f6;
                margin-bottom: 16px;
            ">
                <strong>Description:</strong><br>
                {desc}
            </div>
            """,
            unsafe_allow_html=True,
        )
    docs_files_folder = os.path.join(USE_CASES_FOLDER, current_product, "docs")
    md_content = ""
    if "model_documentation" in selected_data_operation:
        md_path = Path(os.path.join(docs_files_folder, "ModelCard.md"))
        md_content = md_path.read_text(encoding="utf-8")
        st.markdown(md_content)
    if "data_documentation" in selected_data_operation:
        md_path = Path(os.path.join(docs_files_folder, "Datasheet.md"))
        md_content = md_path.read_text(encoding="utf-8")
        st.markdown(md_content)
    for ind, operation in enumerate(operations):
        if operation["type"] == selected_data_operation:
            operation_id = operation["id"]
            operation_name = operation["name"]
            specs = operation["implementation"]["spec"]
            method_name = specs["method_name"]
            inputs = specs["inputs"]
            outputs = specs["outputs"]

            with st.expander(
                f"Operation {operation_id}: {operation_name}", expanded=True
            ):
                cols_oper = st.columns([7, 3])
                with cols_oper[0]:
                    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs(
                        [
                            "Documentation",
                            "Code",
                            "Metadata",
                            "Input",
                            "Output",
                            "Produced artifact",
                        ]
                    )
                    with tab1:
                        st.write("This is the Documentation tab")
                        # val_doc_op = extract_section_by_title(md_content, "Training Procedure")
                        # print(val_doc_op)
                        documentation = st.text_area(
                            "Documentation",
                            key=f"doc_{ind}",
                            value=f"{operation['name']}",  #
                        )
                        # if st.button("Save"):
                        #    md_text = md_text.replace(section_text, documentation)
                        #    md_path.write_text(md_text, encoding="utf-8")
                    with tab2:
                        st.write("This is the Code tab")
                        if current_framework == "dh":
                            method_name = method_name + "_real"
                            step_operations_module = "dh_" + step_operations_module
                        method_content = load_method_content(
                            method_name,
                            current_product,
                            current_framework,
                            step_operations_module,
                        )
                        code = st_ace(
                            value=method_content,
                            language="python",
                            theme="xcode",
                            key=f"code_{ind}",
                            height=300,
                            font_size=14,
                            show_gutter=True,
                            readonly=True,
                        )
                    with tab3:
                        st.write(
                            "This tab contains Metadata specifications for the Inputs/Outputs"
                        )
                        metadata = st_ace(
                            value=json.dumps(specs, indent=2),
                            language="json",
                            theme="xcode",
                            key=f"metadata_{ind}",
                            height=300,
                            font_size=14,
                            show_gutter=True,
                            readonly=True,
                        )
                    with tab4:
                        st.write(
                            "This tab contains Input data that the methods receives"
                        )
                        inputs = st_ace(
                            value=json.dumps(inputs, indent=2),
                            language="json",
                            theme="xcode",
                            key=f"input_{ind}",
                            height=300,
                            font_size=14,
                            show_gutter=True,
                            readonly=True,
                        )
                    with tab5:
                        st.write(
                            "This tab contains Output data that the methods produces"
                        )
                        outputs = st_ace(
                            value=json.dumps(outputs, indent=2),
                            language="json",
                            theme="xcode",
                            key=f"output_{ind}",
                            height=300,
                            font_size=14,
                            show_gutter=True,
                            readonly=True,
                        )
                    with tab6:
                        import streamlit.components.v1 as components

                        for output in eval(outputs):
                            if "report" in output:
                                print(output["report"])
                                report = list(
                                    filter(
                                        lambda x: x["name"] == output["report"],
                                        report_artifacts,
                                    )
                                )
                                if len(report) == 0:
                                    continue
                                report = report[0]
                                report_path = report["config"]["filepath"]
                                report_path = os.path.join(
                                    USE_CASES_FOLDER,
                                    current_product,
                                    "src",
                                    f"{current_framework}_platform",
                                    "artifacts",
                                    "report",
                                    report_path,
                                )
                                if "html" in report_path:
                                    with open(report_path, encoding="utf8") as report_f:
                                        report: Text = report_f.read()
                                        components.html(
                                            report,
                                            width=1000,
                                            height=1200,
                                            scrolling=True,
                                        )

                    if st.button("Run operation", key=f"run_op_{ind}"):
                        run_operation(
                            operation,
                            data_artifacts,
                            model_artifacts,
                            configuration_artifacts,
                            current_product,
                            current_framework,
                            step_operations_module,
                        )
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
                        tools_catalog["ai_operation"] == operation["type"]
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
                                readonly=True,
                            )


def populate_data_artifacts(data_artifacts):
    with st.container():
        st.markdown(
            f"""
                <strong>Raw Input Data:</strong><br>
            """,
            unsafe_allow_html=True,
        )
        for artifact in data_artifacts:
            if "category" in artifact and artifact["category"] == "raw":
                content = ""
                for k, v in artifact.items():
                    content += f"<strong>{k}:</strong> {v}<br>"
                st.markdown(
                    f"""
                    <div style="
                        border: 2px solid #d1d5db;
                        border-radius: 10px;
                        padding: 16px;
                        background-color: #f3f4f6;
                        margin-bottom: 16px;
                    ">
                        {content}
                    </div>
                    """,
                    unsafe_allow_html=True,
                )


def populate_artifacts(artifacts):
    with st.container():
        for artifact in artifacts:
            content = ""
            for k, v in artifact.items():
                content += f"<strong>{k}:</strong> {v}<br>"
            st.markdown(
                f"""
                <div style="
                    border: 2px solid #d1d5db;
                    border-radius: 10px;
                    padding: 16px;
                    background-color: #f3f4f6;
                    margin-bottom: 16px;
                ">
                    {content}
                </div>
                """,
                unsafe_allow_html=True,
            )


def _resolve_vars(specs_list, data_artifacts, config_artifacts, model_artifacts):
    vars = {}
    for item in specs_list:
        artifact_name = list(item.values())[0]
        key = list(item.keys())[0]
        match = next((a for a in data_artifacts if a["name"] == artifact_name), None)
        if match:
            vars[key] = Data(**{k: v for k, v in match.items() if k != "name"})
        match = next((a for a in config_artifacts if a["name"] == artifact_name), None)
        if match:
            vars[key] = Configuration(**{k: v for k, v in match.items() if k != "name"})
        match = next((a for a in model_artifacts if a["name"] == artifact_name), None)
        if match:
            vars[key] = Model(**{k: v for k, v in match.items() if k != "name"})
    return vars


def run_operation(
    operation,
    data_artifacts,
    model_artifacts,
    config_artifacts,
    current_product,
    current_framework="local",
    step_operations_module="data_preparation.py",
):
    product_config_file = os.path.join(
        USE_CASES_FOLDER, current_product, "metadata", f"aipc_{current_framework}.yaml"
    )
    product_operations_file = os.path.join(
        USE_CASES_FOLDER,
        current_product,
        "src",
        f"{current_framework}_platform",
        step_operations_module,
    )
    with open(product_config_file, "r") as yaml_file:
        aipc_configs = yaml.safe_load(yaml_file)
        product_name = aipc_configs["ai_product_name"]
    curr_module = import_from_path("curr_module", product_operations_file)

    specs = operation["implementation"]["spec"]
    method_name = specs["method_name"]

    input_vars = _resolve_vars(
        specs["inputs"], data_artifacts, config_artifacts, model_artifacts
    )
    input_vars.update(
        _resolve_vars(
            specs["outputs"], data_artifacts, config_artifacts, model_artifacts
        )
    )
    print(input_vars)

    func = getattr(curr_module, method_name)
    if current_framework == "local":
        func(**input_vars)
    else:
        func(product_name, **input_vars)
    # method_name = globals()[method_name]
    # result = method_name(**input_vars)


def main():
    st.set_page_config(layout="wide", page_title="Guided AI Product")
    # st.sidebar.title("AI product development lifecycle")
    current_framework = st.sidebar.selectbox(
        "Tool governance platform", platform_list, index=0
    )

    use_cases_list_names = []
    for product in use_cases_list:
        if "metadata" in os.listdir(os.path.join(USE_CASES_FOLDER, product)):
            config_file = os.path.join(
                USE_CASES_FOLDER, product, "metadata", "aipc_local.yaml"
            )
            yaml_path = Path(config_file)
            yaml_text = yaml_path.read_text()
            with open(config_file, "r") as yaml_file:
                aipc_configs = yaml.safe_load(yaml_file)
            use_cases_list_names.append(
                f"{product}: ({aipc_configs['ai_product_name']})"
            )
    current_product_label = st.sidebar.selectbox(
        "AI Products list", use_cases_list_names
    )
    current_product = current_product_label.split(": ")[0]

    session_state_params(current_product, current_framework)

    with st.sidebar:
        current_step = option_menu(
            "AI lifecycle stages",
            ["Data preparation", "Modeling", "Operationalisation"],
            icons=["graph-up", "tools", "speedometer2"],
            menu_icon="sliders",
            default_index=0,
            orientation="vertical",
        )
        selected_aspect = option_menu(
            "AI system requirements",
            ["Baseline", "Fairness", "Robustness"],
            icons=["house", "list-task", "gear"],
            menu_icon="cast",
            default_index=0,
            orientation="vertical",
        )
    if "view" not in st.session_state:
        st.session_state.view = "Operations Visualisation"
    col1, col2, col3 = st.columns(3)
    with col1:
        if st.button("Operations Visualisation"):
            st.session_state.view = "Operations Visualisation"
    with col2:
        if st.button("Declarative Configuration Visualisation"):
            st.session_state.view = "Declarative Configuration"
    with col3:
        if st.button("Artifacts Visualisation"):
            st.session_state.view = "Artifacts Visualisation"

    if st.session_state.view == "Operations Visualisation":
        show_stage(current_step, selected_aspect, current_product, current_framework)
    elif st.session_state.view == "Artifacts Visualisation":
        show_artifacts(
            current_step, selected_aspect, current_product, current_framework
        )
    else:
        if "metadata" in os.listdir(os.path.join(USE_CASES_FOLDER, current_product)):
            config_file = os.path.join(
                USE_CASES_FOLDER, current_product, "metadata", "aipc_local.yaml"
            )
            yaml_path = Path(config_file)
            yaml_text = yaml_path.read_text()
        st.code(yaml_text, language="yaml")


if __name__ == "__main__":
    main()
