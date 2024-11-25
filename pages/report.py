"""
This is the Data page, where you can view the dataset and its information.

"""

###################################### import ######################################

# library
from sidebar import set_report
import streamlit as st
import pandas as pd
import numpy as np
import gettext
import os



from functions.overview_functions import show_overview_table, show_alerts, show_dtype_df, missing_values_plot, show_outliers
from functions.visualization_functions import categorical_format, numerical_format, timeseries, string
from functions.correlation_functions import get_pairplot, normality_test, plot_heatmap
from functions.modeling_functions import show_all_results

###################################### set  ######################################

# set the current page context
st.session_state.current_page = "Report"

# language setting
locale_path = os.path.join(os.path.dirname(__file__), 'locales')
translator = gettext.translation('base', localedir=locale_path, languages=[st.session_state.language], fallback=True)
translator.install()

# sidebar
set_report()

##################################### functions ######################################

def categorical(df, dtype_df):
    cat_col = dtype_df[dtype_df['Data Type'].isin(['Categorical'])].index.to_list()
    df_cat = df[cat_col]

    st.markdown("### Categorical")
    with st.container(border=True):
        categorical_format(df_cat, True)

def boolean(df, dtype_df):
    bool_col = dtype_df[dtype_df['Data Type'].isin(['Boolean'])].index.to_list()
    df_bool = df[bool_col]

    st.markdown("### Boolean")
    with st.container(border=True):
        categorical_format(df_bool, False)

def continuous(df, dtype_df):
    continuous = dtype_df[dtype_df['Data Type'].isin(['Numeric (Continuous)'])].index.to_list()
    df_con = df[continuous]

    st.markdown("### Continuous")
    with st.container(border=True):
        numerical_format(df_con, False)

def discrete(df, dtype_df):
    discrete = dtype_df[dtype_df['Data Type'].isin(['Numeric (Discrete)'])].index.to_list()
    df_dis = df[discrete]

    st.markdown("### Discrete")
    with st.container(border=True):
        numerical_format(df_dis, True)

def draw_pairplot(df, target):
    st.markdown("### Pairplot")

    with st.container(border=True):
        if target not in df.columns:
            pairplot_fig = get_pairplot(df, None)
        else:
            pairplot_fig = get_pairplot(df, target)
        st.pyplot(pairplot_fig)

def correlation_heatmap(df):
    st.markdown("### Correlation Heatmap")
    with st.container(border=True):
        tab1, tab2, tab3 = st.tabs(['Pearson', 'Spearman', 'Kendall'])

        # pearson
        with tab1:
            plot_heatmap(df, 'pearson')
            
        # spearman
        with tab2:
            plot_heatmap(df, 'spearman')

        # kendall
        with tab3:
            plot_heatmap(df, 'kendall')

##################################### report ######################################

# report page
# report page
def report_page():
    st.title("Report")

    if 'df' in st.session_state:
        page = st.session_state.report_page
        df = st.session_state.df
        target = st.session_state.target
        dtype_df = st.session_state.dtype_df
        numeric_col = dtype_df[dtype_df['Data Type'].isin(['Numeric (Discrete)', 'Numeric (Continuous)'])].index.to_list()

        if page is not None:
            # Select 열이 True인 페이지만 필터링
            selected_pages = page[page["Select"]]
            
            for _, row in selected_pages.iterrows():
                if row['Page'] == "Overview":
                    show_overview_table(df, dtype_df)
                elif row['Page'] == "Alert":
                    show_alerts(df)
                elif row['Page'] == "Data Types":
                    show_dtype_df(dtype_df)
                elif row['Page'] == "Missing Values":
                    missing_values_plot(df)
                elif row['Page'] == "Outlier":
                    show_outliers(df, numeric_col)
                elif row['Page'] == "Categorical":
                    categorical(df, dtype_df)
                elif row['Page'] == "Boolean":
                    boolean(df, dtype_df)
                elif row['Page'] == "Discrete":
                    discrete(df, dtype_df)
                elif row['Page'] == "Continuous":
                    continuous(df, dtype_df)
                elif row['Page'] == "Time Series":
                    time_col = dtype_df[dtype_df['Data Type'].isin(['Time'])].index.to_list()
                    timeseries(df, time_col, numeric_col)
                elif row['Page'] == "String":
                    str_col = dtype_df[dtype_df['Data Type'].isin(['String'])].index.to_list()
                    string(df, str_col)
                elif row['Page'] == "Pairplot":
                    draw_pairplot(df[numeric_col], target)
                elif row['Page'] == "Normality":
                    st.markdown("### Normality Test")
                    normality_test(df[numeric_col])
                elif row['Page'] == "Correlation Heatmap":
                    correlation_heatmap(df)
                elif row['Page'] == "Modeling":
                    st.markdown("### Results")
                    with st.container(border=True):
                        show_all_results(st.session_state.model_type)
    else:
        st.warning(_("Please upload a CSV file to view this page."))

##################################### main ######################################
# main page
report_page()

