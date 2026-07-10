# streamlit: page_name = "New AI Product"
import os
import yaml
import asyncio
import pandas as pd
import streamlit as st
from utils import populate_stages, get_application_domains
from dotenv import load_dotenv, find_dotenv

from langchain_mcp_adapters.client import MultiServerMCPClient
from langchain_mcp_adapters.tools import load_mcp_tools
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, MessagesState, START, END
from langgraph.prebuilt import ToolNode

_ = load_dotenv(find_dotenv())

st.set_page_config(layout="wide", page_title="New AI Product")  # , page_icon="📊"
st.title("Generate a new AI Product: guide through the steps")
# st.sidebar.header("DataFrame Demo")

current_folder = os.path.dirname(os.path.abspath(__file__))
parent_folder = os.path.dirname(current_folder)
parent_folder = os.path.dirname(parent_folder)
pipeline_definitions_folder = os.path.join(
    parent_folder, "framework/temlops/config/pipeline_definitions.yaml"
)

# Step 1: definition of the necessary operations (active/inactive ops)
with open(pipeline_definitions_folder, "r") as yaml_file:
    pipeline_configs = yaml.safe_load(yaml_file)
    requirements_dimensions = list(map(lambda x: x.capitalize(), pipeline_configs["requirements_dimensions"]))

def lifecycle_stages():
    domain_col, task_col, use_type_col = st.columns(3)
    st.write(get_application_domains())
    with domain_col:
        application_domain = st.selectbox(
            "Application Domain",
            get_application_domains(),
        )
    with task_col:
        ai_task = st.selectbox(
            "AI Task",
            ["Classification", "Regression", "Clustering", "Object Detection", "Text Generation", "Anomaly Detection", "Recommendation"],
        )
    with use_type_col:
        ai_type_of_use = st.selectbox(
            "AI Type of Use",
            ["Decision Support", "Automation", "Augmentation", "Monitoring", "Prediction", "Generation", "Personalization"],
        )
    st.session_state["application_domain"] = application_domain
    st.session_state["ai_task"] = ai_task
    st.session_state["ai_type_of_use"] = ai_type_of_use

    name_col, desc_col = st.columns(2)
    with name_col:
        ai_prod_name = st.text_area("AI product name")
    with desc_col:
        ai_prod_desc = st.text_area("AI product description", value="""Generate code implementation that is able to
                                implement data drift detection for a given dataset """)
    st.write("AI system's stages and operations")
    selected_operations = populate_stages(pipeline_configs, createview=True)
    st.session_state["selected_operations"] = selected_operations
    return ai_prod_name, ai_prod_desc


# Step 2: definition of the requirements dimensions to be satisfied according to the AI product design objectives
def show_new_prod_skeleton():
    st.session_state["page"] = "new_prod"
    checklist_requirements = st.multiselect(
        "AI system's requirements dimensions",
        requirements_dimensions,
        ["Baseline", "Robustness"],
    )


# Step 3: specification of the data artifacts and the AI product objectives (classification, clustering, information extraction)
# subtasks:
#   - load data artifact from corresponding dh project
#   - load data characteristics into the prompt context in order to facilitate the planning of the operations
#data_atifact = pd.read_csv("artifacts/data/data.csv") # TODO


# Step 4: CoT prompting to plan the operations of a new AI product
# subtasks:
#   - select the right open source toolkits to use for implementing each operation according to the pre-defined requirements
#   - generate code snippets based on the selected toolkits for each operation
async def run_mcp_query(user_input):   
    # Model
    model = ChatOpenAI(model="gpt-5") #"gpt-5"

    # MCP Client via HTTP
    client = MultiServerMCPClient(
        {
            "mlops_tai_engineers": {
                "transport": "streamable_http",
                "url": "http://127.0.0.1:8082/mcp"  
            },
            "file_system": {
                "transport": "streamable_http",
                "url": "http://127.0.0.1:8080/mcp"  
            }
        }
    )
    tools = await client.get_tools() # await load_mcp_tools(client) #
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
    result = await graph.ainvoke({"messages": [{"role": "user", "content": user_input}]})

    # Extract last message text
    last_msg = result["messages"][-1].content
    return last_msg if isinstance(last_msg, str) else str(last_msg)



# Step 5: generate the new AI product folder structure with the necessary code files and configuration files
def generate_prod_action(ai_prod_desc):
    if st.button("Create AI product skeleton", key=f"generate_product"):        
        template_aipc_folder = os.path.join(parent_folder, "framework/temlops/aipc_template")
        new_prod_folder = os.path.join(parent_folder, "framework/temlops/use_cases/new_prod")
        folder = f" FOLDER: {new_prod_folder}"
        os.makedirs(new_prod_folder, exist_ok=True)
        with st.spinner("Thinking..."):
            selected_ops = st.session_state["selected_operations"]
            selected_operations = [op for op, selected in selected_ops.items() if selected]
            plan_prompt = open(f"guided_ui/pages/plan.md", "r").read()
            ai_prod_desc = f"{plan_prompt}. \n\n Copy recursively the files and subdirectories inside the folder {template_aipc_folder} into the new AI product folder {new_prod_folder}.  \n\n  The selected operations are: {selected_operations}"
            print(ai_prod_desc)
            answer = asyncio.run(run_mcp_query(ai_prod_desc ))
            st.success("Operation completed successfully!")
            st.success(answer)               
        
    

if __name__ == "__main__":
    ai_prod_name, ai_prod_desc = lifecycle_stages()
    show_new_prod_skeleton()
    generate_prod_action(ai_prod_desc)
    
