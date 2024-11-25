import streamlit as st
import os
import gettext
from functions.correlation_functions import pairplot, scatter, heatmap

###################################### set  ######################################

# set the current page context
st.session_state.current_page = "Correlation"

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

##################################### Correlation Page ######################################

def correlation_page():
    # title
    st.title("Correlation")

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
            numeric_col = dtype_df[dtype_df['Data Type'].isin(['Numeric (Continuous)', 'Numeric (Discrete)'])].index.to_list()

            tab1, tab2, tab3 = st.tabs(['Pairplot', 'Scatter Plot', 'Correlation Heatmap'])

            # pariplot
            with tab1:
                pairplot(df[numeric_col])
            
            # scatter plot
            with tab2:
                scatter(df[numeric_col])
            
            # heatmap
            with tab3:
                heatmap(df[numeric_col])
        
        else:
            st.warning(_("Please upload a CSV file to view this page."))
    else:
        st.error(_("Please enter OpenAI API Key"))   
         
##################################### main ######################################
# main page
correlation_page()