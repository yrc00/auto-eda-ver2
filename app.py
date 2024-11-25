"""

This is the main page

"""

###################################### import ######################################

# library
import gettext
import os

import streamlit as st
import pandas as pd

# sidebar
from sidebar import slm_menu, llm_menu, language_selector, llm_model_selector, llm_response

###################################### set ######################################

# set the current page context
st.session_state.current_page = "Home"

# language setting
if "language" not in st.session_state:
    st.session_state.language="en"

locale_path = os.path.join(os.path.dirname(__file__), 'locales')
translator = gettext.translation('base', localedir=locale_path, languages=[st.session_state.language], fallback=True)
translator.install()
_ = translator.gettext

# llm model setting
if "llm_model" not in st.session_state:
    st.session_state.llm_model="Phi3"

# view llm analysis
if "llm_response" not in st.session_state:
    st.session_state.llm_response = True

st.session_state.api_error = False

if "openai_api_key" not in st.session_state:
    st.session_state.openai_api_key = ""

if "apply_btn" not in st.session_state:
    st.session_state.apply_btn = False

###################################### home_page ######################################

def home_page():
    st.title(_("Welcome to Auto EDA!"))
    st.markdown(_("This is an Auto EDA Website developed by Yerim Choi and Yujin Min"))
    st.markdown(_("You can find the source code [here](https://github.com/yrc00/auto-eda/tree/main)"))

    st.markdown(_("### **Instructions**"))
    st.markdown(_("- To use Auto EDA, upload your csv file to the Data Page."))
    st.markdown(_("- Navigate to different pages using the Sidebar."))
    st.markdown(_("- View Data, Overview, Visualization, Correlation, Modeling, and Chatbot pages."))
    st.markdown(_("- Use Chatbot to ask questions and gain insights about your data."))
    st.markdown(_("**Note**: This is a demo website so it may not work as expected."))

    st.markdown(_("### **About Pages**"))
    st.markdown(_("- **Data**: Upload csv file and set up dataset."))
    st.markdown(_("- **Overview**: View the dataset and its information."))
    st.markdown(_("- **Visualization**: Visualize the dataset using a variety of plots."))
    st.markdown(_("- **Correlation**: Check correlation between different columns."))
    st.markdown(_("- **Modeling**: Perform supervised and unsupervised modeling."))
    st.markdown(_("- **Chatbot**: Ask questions and gain insights about the dataset."))

###################################### main ######################################

if __name__ == "__main__":
    # Get the current page
    if st.session_state.llm_model in ["gpt-3.5-turbo", "gpt-4", "gpt-4o", "gpt-4o-mini"]:
        selected_page = llm_menu()
    else:
        selected_page = slm_menu()

    # Set the current page in session state
    if selected_page is not None:
        st.session_state.current_page = selected_page.title

    # Render the selected page
    if st.session_state.current_page == "Home":
        home_page()

    # sidebar
    language_selector()
    llm_model_selector()
    llm_response()
