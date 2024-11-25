"""

This is the sidebar page

"""

###################################### import ######################################

# library
import streamlit as st
import pandas as pd
import os

###################################### menu ######################################

# print menu
def slm_menu():
    pages = {
        "Auto EDA": [
            st.Page("app.py", title="Home", icon="🏠"),
        ],
        "Report": [
            st.Page("pages/data.py", title="Data", icon="💾"),
            st.Page("pages/overview.py", title="Overview", icon="🔍"),
            st.Page("pages/visualization.py", title="Visualization", icon="📊"),
            st.Page("pages/correlation.py", title="Correlation", icon="🔗"),
            st.Page("pages/modeling.py", title="Modeling", icon="🤖"),
            st.Page("pages/report.py", title="Report", icon="📝"),
        ],
    }
    pg = st.navigation(pages)
    return pg.run()

def llm_menu():
    pages = {
        "Auto EDA": [
            st.Page("app.py", title="Home", icon="🏠"),
        ],
        "Report": [
            st.Page("pages/data.py", title="Data", icon="💾"),
            st.Page("pages/overview.py", title="Overview", icon="🔍"),
            st.Page("pages/visualization.py", title="Visualization", icon="📊"),
            st.Page("pages/correlation.py", title="Correlation", icon="🔗"),
            st.Page("pages/modeling.py", title="Modeling", icon="🤖"),
            st.Page("pages/report.py", title="Report", icon="📝"),
        ],
        "Chatbot": [
            st.Page("pages/chatbot.py", title="Chatbot", icon="💬"),
        ]

    }
    pg = st.navigation(pages)
    return pg.run()
    

###################################### language selection ######################################

# set language
def language_selector():
    # language list
    languages = ["en", "ko"]

    # set language
    selected_language = st.sidebar.selectbox("Select Language", options=languages)
    if selected_language != st.session_state.language:
        st.session_state.language = selected_language
        st.rerun()

###################################### model selection ######################################

def llm_model_selector():
    # LLM model list
    models = ["Phi3", "gpt-4", "gpt-3.5-turbo", "gpt-4o", "gpt-4o-mini"]

    # Set default model to "Phi3" if not already set
    if "llm_model" not in st.session_state:
        st.session_state.llm_model = "Phi3"

    # Sidebar selection for the LLM model
    selected_model = st.sidebar.selectbox(
        "Select LLM Model:",
        options=models,
        index=models.index(st.session_state.llm_model),
    )

    # Update the selected model
    if selected_model != st.session_state.llm_model:
        st.session_state.llm_model = selected_model
        st.rerun()

    ###################### if you want to use secrets.toml comment out below ######################
    # Show API key input fields based on the selected model
    if selected_model in ["gpt-3.5-turbo", "gpt-4", "gpt-4o", "gpt-4o-mini"]:
        # OpenAI API key input
        openai_api_key = st.sidebar.text_input(
            "Enter OpenAI API Key:",
            type="password",
            value=st.session_state.get("openai_api_key", ""),
        )

        # Update OpenAI API key
        if openai_api_key != st.session_state.get("openai_api_key", ""):
            st.session_state.openai_api_key = openai_api_key
            os.environ['OPENAI_API_KEY'] = openai_api_key
            st.sidebar.success("OpenAI API Key updated successfully!")
    # else:
    #     # HuggingFace API key input
    #     huggingface_api_key = st.sidebar.text_input(
    #         "Enter HuggingFace API Key:",
    #         type="password",
    #         value=st.session_state.get("huggingface_api_key", ""),
    #     )

    #     # Update HuggingFace API key
    #     if huggingface_api_key != st.session_state.get("huggingface_api_key", ""):
    #         st.session_state.huggingface_api_key = huggingface_api_key
    #         os.environ['HUGGINGFACEHUB_API_TOKEN'] = huggingface_api_key
    #         st.sidebar.success("HuggingFace API Key updated successfully!")

###################################### llm response ######################################

# llm response
def llm_response():
    view = st.sidebar.checkbox("View LLM Analysis ", value=True)
    if view != st.session_state.llm_response:
        st.session_state.llm_response = view
        st.rerun()

###################################### report selection ######################################

# set report page
def set_report():
    contents = ["Overview", "Alert", "Data Types", "Missing Values", "Outlier", "Categorical", 
                "Boolean", "Discrete", "Continuous", "Time Series", "String", 
                "Pairplot", "Normality", "Correlation Heatmap", "Modeling"]

    if 'report_page' not in st.session_state:
        st.session_state.report_page = pd.DataFrame({
            "Page": contents,
            "Select": [False] * len(contents)
        })

    with st.sidebar:
        st.markdown("**Select the page you want to add**")
        edited_df = st.data_editor(
            st.session_state.report_page,
            column_config={
                "Select": st.column_config.CheckboxColumn(
                    label="Select",
                    help="Check to include this page"
                )
            },
            key="page_selector"
        )

        # Update session state only if data changes
        if not edited_df.equals(st.session_state.report_page):
            st.session_state.report_page = edited_df
            st.rerun()

