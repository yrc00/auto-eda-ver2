import streamlit as st
import os
import gettext

from functions.modeling_functions import supervised, clustering, pca

###################################### set  ######################################

# set the current page context
st.session_state.current_page = "Modeling"

# language setting
locale_path = os.path.join(os.path.dirname(__file__), 'locales')
translator = gettext.translation('base', localedir=locale_path, languages=[st.session_state.language], fallback=True)
translator.install()
_ = translator.gettext

##################################### Modeling Page ######################################

# modeling page
def modeling_page():
    # title
    st.title("Modeling")

    if 'df' in st.session_state:
        df = st.session_state.df
        dtype_df = st.session_state.dtype_df
        target = st.session_state.target
        numeric_col = dtype_df[dtype_df['Data Type'].isin(['Numeric (Continuous)', 'Numeric (Discrete)', 'Boolean'])].index.to_list()

        tab1, tab2, tab3 = st.tabs(['Supervised', 'Clustering', 'PCA'])

        # supervised
        with tab1:
            supervised(df.dropna(), target)
        
        with tab2:
            clustering(df[numeric_col], target)
        
        with tab3:
            pca(df[numeric_col], target)
    
    else:
        st.warning(_("Please upload a CSV file to view this page."))

##################################### main ######################################
# main page
modeling_page()