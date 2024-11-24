"""

This is the sidebar page

"""

###################################### import ######################################

# library
import streamlit as st
import pandas as pd

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

    # default model = Phi3
    if "llm_model" not in st.session_state:
        st.session_state.llm_model = "Phi3"

    # select model from sidebar
    selected_model = st.sidebar.selectbox(
        "Select LLM Model",
        options=models,
        index=models.index(st.session_state.llm_model),
    )

    # update model
    if selected_model != st.session_state.llm_model:
        st.session_state.llm_model = selected_model
        st.rerun()
        
    # OpenAI API Key input
    if selected_model in ["gpt-3.5-turbo", "gpt-4", "gpt-4o", "gpt-4o-mini"]:
        # API 키 입력 필드
        openai_api_key = st.sidebar.text_input(
            "Enter OpenAI API Key:",
            type="password",
            value=st.session_state.get("openai_api_key", ""),
        )

        # API key update
        if openai_api_key != st.session_state.get("openai_api_key", ""):
            st.session_state.openai_api_key = openai_api_key
            st.sidebar.success("API Key updated successfully!")


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
    pages = ["Overview", "Alert","Data Types","Missing Values", "Outlier", "Categorical", "Boolean", "Discrete", 
         "Continuous", "Time Series", "String", "Pairplot", "Normality", "Correlation Heatmap", "Modeling"]

    with st.sidebar:
        # checkbox to select pages
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

        # Check if edited_df is different from current report_page
        if not edited_df.equals(st.session_state.report_page):
            st.session_state.report_page = edited_df
            st.rerun()

