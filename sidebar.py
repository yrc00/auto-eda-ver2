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

# # set llm model
# def llm_model_selector():
#     # List of available models
#     models = ["Phi3", "gpt-4", "gpt-3.5-turbo", "gpt-4o", "gpt-4o-mini"]
    
#     # Retrieve the selected model from session state if it exists, else set to default "Phi3"
#     selected_model = st.sidebar.selectbox(
#         "Select LLM Model", 
#         options=models, 
#         index=models.index(st.session_state.get('llm_model', 'Phi3'))
#     )
    
#     # If a new model is selected, update the session state and trigger rerun
#     if selected_model != st.session_state.get('llm_model', 'Phi3'):
#         st.session_state.llm_model = selected_model
#         # Trigger rerun to apply the model change
#         st.rerun()
    
#     # Only show OpenAI API Key input if the model is "gpt-4o" or "gpt-4o-mini"
#     if selected_model in ["gpt-3.5-turbo", "gpt-4", "gpt-4o", "gpt-4o-mini"]:        
#         # OpenAI API Key input
#         openai_api_key = st.sidebar.text_input(
#             "Enter OpenAI API Key:",
#             type="password",
#             value=st.session_state.openai_api_key
#         )
        
#         # Update API key in session state if entered
#         if st.session_state.openai_api_key is "" and openai_api_key:
#             st.session_state.openai_api_key = openai_api_key
#             st.sidebar.success("OpenAI API Key is set.")
#             st.rerun()
#         elif openai_api_key != st.session_state.openai_api_key:  # Missing colon fixed here
#             st.session_state.openai_api_key = openai_api_key
#             st.sidebar.success("OpenAI API Key is set.")
#             st.rerun()
#         else:
#             st.sidebar.error("Please enter OpenAI API Key")
#             # Clear API key if no input is given
#             if 'openai_api_key' in st.session_state:
#                 del st.session_state['openai_api_key']
#                 # Trigger rerun after clearing the key
#                 st.rerun()

def llm_model_selector():
    # LLM 모델 리스트
    models = ["Phi3", "gpt-4", "gpt-3.5-turbo", "gpt-4o", "gpt-4o-mini"]

    # 현재 선택된 모델 상태 가져오기 (없으면 기본값 'Phi3' 사용)
    if "llm_model" not in st.session_state:
        st.session_state.llm_model = "Phi3"

    # 사이드바에서 모델 선택
    selected_model = st.sidebar.selectbox(
        "Select LLM Model",
        options=models,
        index=models.index(st.session_state.llm_model),
    )

    # 모델이 변경되었을 경우 상태 업데이트 및 즉시 반영
    if selected_model != st.session_state.llm_model:
        st.session_state.llm_model = selected_model
        st.rerun()  # 새로고침
        
    # OpenAI API Key 입력 (특정 모델에 한정)
    if selected_model in ["gpt-3.5-turbo", "gpt-4", "gpt-4o", "gpt-4o-mini"]:
        # API 키 입력 필드
        openai_api_key = st.sidebar.text_input(
            "Enter OpenAI API Key:",
            type="password",
            value=st.session_state.get("openai_api_key", ""),
        )

        # API 키가 변경되었으면 상태 업데이트 및 반영
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

