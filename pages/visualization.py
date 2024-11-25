import streamlit as st
import os
import gettext
from functions.visualization_functions import categorical, numerical, timeseries, string

###################################### set ######################################

# set the current page context
st.session_state.current_page = "Visualization"

# language setting
locale_path = os.path.join(os.path.dirname(__file__), 'locales')
translator = gettext.translation('base', localedir=locale_path, languages=[st.session_state.language], fallback=True)
translator.install()
_ = translator.gettext

# load the environment variables
if "llm_model" not in st.session_state:
    st.session_state.llm_model = "Phi3"

#################### get API key from secrets.toml ####################

# HuggingFace API token
huggingface_api_token = st.secrets.get("HUGGINGFACEHUB_API_TOKEN")

if huggingface_api_token:
    os.environ['HUGGINGFACEHUB_API_TOKEN'] = huggingface_api_token
    api_key = huggingface_api_token
else:
    st.error(_("HUGGINGFACEHUB_API_TOKEN is missing. Please check your secrets.toml file."))
    api_key = ""

# # OpenAI API key
# openai_api_key = st.secrets.get("OPENAI_API_KEY")

# if openai_api_key:
#     os.environ['OPENAI_API_KEY'] = openai_api_key
#     api_key = openai_api_key
# else:
#     st.error(_("OPENAI_API_KEY is missing. Please check your secrets.toml file."))
#     api_key = ""

#################### get API key from sidebar ####################

# huggingface API token
if "llm_model" in st.session_state:
    if st.session_state.llm_model in ["gpt-3.5-turbo", "gpt-4", "gpt-4o", "gpt-4o-mini"]:
        api_key = st.session_state.get("openai_api_key", "")
    # else:
    #     api_key = st.session_state.get("huggingface_api_key", "")
        
###################################### visualization page ######################################

def visualization_page():
    # title
    st.title("Visualization")

    if not st.session_state.llm_response:
        view_page = True
    elif api_key is "":
        view_page = False
    else:
        view_page = True

    if view_page:
        if 'df' in st.session_state:
            df = st.session_state.df
            dtype_df = st.session_state.dtype_df

            cat_col = dtype_df[dtype_df['Data Type'].isin(['Categorical'])].index.to_list()
            bool_col = dtype_df[dtype_df['Data Type'].isin(['Boolean'])].index.to_list()
            num_col = dtype_df[dtype_df['Data Type'].isin(['Numeric (Discrete)', 'Numeric (Continuous)'])].index.to_list()
            discrete = dtype_df[dtype_df['Data Type'].isin(['Numeric (Discrete)'])].index.to_list()
            continuous = dtype_df[dtype_df['Data Type'].isin(['Numeric (Continuous)'])].index.to_list()
            time_col = dtype_df[dtype_df['Data Type'].isin(['Datetime'])].index.to_list()
            str_col = dtype_df[dtype_df['Data Type'].isin(['String'])].index.to_list()
        
            tab1, tab2, tab3, tab4 = st.tabs(['Categorical', 'Numerical', 'Time Series', 'String'])

            with tab1:
                categorical(df[cat_col], df[bool_col])

            with tab2:
                numerical(df[discrete], df[continuous])

            with tab3:
                timeseries(df, time_col, num_col)

            with tab4:
                string(df, str_col)

        else:
            st.warning(_("Please upload a CSV file to view this page."))
    else:
        st.error(_("Please enter OpenAI API Key"))   

###################################### main ######################################
# main
visualization_page()