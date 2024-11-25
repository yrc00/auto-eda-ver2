"""
This is the Overview page

"""

###################################### import ######################################

# library
from dotenv import load_dotenv
import gettext
import os

import streamlit as st
import pandas as pd
import numpy as np

from st_mui_table import st_mui_table
import missingno as msno
import matplotlib.pyplot as plt
import seaborn as sns

import re
from langchain import LLMChain
from langchain.prompts import PromptTemplate
from langchain.llms import HuggingFaceHub
from deep_translator import GoogleTranslator
import textwrap
import openai
import time

###################################### set  ######################################

# language setting
locale_path = os.path.join(os.path.dirname(__file__), 'locales')
translator = gettext.translation('base', localedir=locale_path, languages=[st.session_state.language], fallback=True)
translator.install()
_ = translator.gettext

##################################### google translate ######################################

# clean response from LLM
def remove_duplicate_sentences(text):
    # split text into sentences
    sentences = text.split('.')
    # remove duplicate sentences
    unique_sentences = list(dict.fromkeys(sentences))
    cleaned_text = '.'.join(unique_sentences).strip()
    
    return cleaned_text

def extract_cleaned_response(response_text):
    # find the start of the response text
    delimiters = ["Answer:", "## Your task", "Input:", "## Response", "B:"]
    response_start = len(response_text)
    for delimiter in delimiters:
        temp_start = response_text.find(delimiter)
        if temp_start != -1 and temp_start < response_start:
            response_start = temp_start + len(delimiter)

    # text after "Answer:" is the response
    response_text = response_text[response_start:].strip()

    # delete text after the next delimiter
    for delimiter in delimiters:
        delimiter_start = response_text.find(delimiter)
        if delimiter_start != -1:
            response_text = response_text[:delimiter_start].strip()
    
    # delete duplicate sentences
    response_text = remove_duplicate_sentences(response_text)

    # delete last sentence after period
    last_period = response_text.rfind(".")
    if last_period != -1:
        response_text = response_text[:last_period + 1].strip()

    return response_text

# google translate
def google_translate(text, language):
    # Define chunk size to be safe (less than 5000, e.g., 4000 characters)
    max_chunk_length = 4000
    translator = GoogleTranslator(source='auto', target=language)
    
    # Split the text into chunks of 4000 characters
    split_text = [text[i:i + max_chunk_length] for i in range(0, len(text), max_chunk_length)]
    
    # Translate each chunk and concatenate the results
    translated_text = ""
    for text_chunk in split_text:
        translated_text += translator.translate(text_chunk) + " "
    
    return translated_text.strip()

##################################### Overview ######################################

# find zero values
def is_zero(series):
    return series == 0

# show table
def show_table(df, key, n=None):
    st_mui_table(
        df,
        enablePagination=False,
        customCss="",  
        paginationSizes=[n] if n else [],
        size="small",
        padding="normal",
        showHeaders=False,
        key=key,
        stickyHeader=False,
        paperStyle={ 
            "width": '100%',  
            "overflow": 'auto',
            "paddingBottom": '1px', 
            "border": '1px solid rgba(224, 224, 224, 1)'
        },
        detailColumns=[],
        detailColNum=1,
        detailsHeader="Details",
        showIndex=False
    )

# overview table
def gen_overview_table(df, dtype_df):
    # rows, columns
    rows, columns = df.shape

    # missing cells
    missing = df.isna().sum().sum()
    missing_per = missing / (rows * columns) * 100

    # duplicate rows
    duplicate = df.duplicated().sum()
    duplicate_per = df.duplicated().mean() * 100

    # memory usage
    memory = df.memory_usage().sum() / 1024
    memory_per = df.memory_usage().mean()

    st.session_state.overview_table = pd.DataFrame({
        'Metric': [
            'Number of Columns', 'Number of Rows', 'Missing Cells', 'Missing Cells (%)', 
            'Duplicate Rows', 'Duplicate Rows (%)', 'Total Size in Memory (KB)', 
            'Average Size in Memory (B)'
        ],
        'Value': [
            f'{columns}',  
            f'{rows}',  
            f'{missing}',  
            f'{missing_per:.2f}%',  
            f'{duplicate}',  
            f'{duplicate_per:.2f}%',  
            f'{memory:.2f} KB',  
            f'{memory_per:.2f} B',
        ]
    })

    # data types
    dtype_table = pd.DataFrame(dtype_df['Data Type'].value_counts())
    dtype_table = dtype_table.reset_index()
    dtype_table.columns = ['Data Type', 'Count']
    st.session_state.dtype_table = dtype_table

def show_overview_table(df, dtype_df):
    st.markdown("### Overview")

    gen_overview_table(df, dtype_df)

    # overview table
    overview_table = st.session_state.overview_table
    overview_add = pd.DataFrame({'Metric': [''], 'Value': ['']})
    overview_table= pd.concat([overview_table, overview_add], ignore_index=True)

    # data type table
    dtype_table = st.session_state.dtype_table
    dtype_add = pd.DataFrame({'Data Type': ['Additional'], 'Count': ['']})
    dtype_table = pd.concat([dtype_table, dtype_add], ignore_index=True)

    # show tables
    with st.container(border=True):
        col1, col2 = st.columns(2)

        # show overview table
        with col1:
            st.markdown("**Dataset Overview**")
            show_table(overview_table, key="overview_table", n=8)
        
        # show data type table
        with col2:
            st.markdown("**Data Types**")
            show_table(dtype_table, key="dtype_table", n=dtype_table.shape[0])

# print alerts
def show_alerts(df):
    with st.expander("**Alerts**"):
        # duplicate rows
        duplicate = df.duplicated().sum()
        duplicate_per = df.duplicated().mean() * 100

        col1, col2 = st.columns([3, 1])

        if duplicate > 0:
            with col1:
                st.write(f"Dataset has {duplicate} ({duplicate_per:.2f}%) duplicated rows")
            with col2:
                st.markdown(":gray-background[Duplicated]")

        # missing values, zeros
        for col in df.columns:
            # missing values
            col_missing = df[col].isna().sum()
            col_missing_per = df[col].isna().mean() * 100
            if col_missing > 0:
                with col1:
                    st.write(f"```{col}``` has {col_missing} ({col_missing_per:.2f}%) missing values")
                with col2:
                    st.markdown(":blue-background[Missing]")
                
            # zero values
            col_zeros = is_zero(df[col]).sum()
            col_zeros_per = is_zero(df[col]).mean() * 100
            if col_zeros > 0:
                with col1:
                    st.write(f"```{col}``` has {col_zeros} ({col_zeros_per:.2f}%) zero values")
                with col2:
                    st.markdown(":green-background[Zeros]")

def show_dtype_df(dtype_df):
    # show data types
    with st.expander("**Datatype of Columns**"):
        st.table(dtype_df)

# Function to get data type description
def get_datatype_description(dtype):
    datatype_descriptions = {
        "Categorical": "The data type dividing data into several categories or items",
        "Numeric (Discrete)": "The data type that can be measured and displayed numerically and can be expressed as clear values, such as integers",
        "Numeric (Continuous)": "The data type that can be measured and displayed numerically and can represent a value between a certain interval, such as a real number",
        "Datetime": "The data type representing time series data",
        "Boolean": "The data type indicating true or false values",
        "String": "The data type representing text.",
        "Other": "The data type not included in Category, Numerical, Boolean, String, Datetime"
    }

    return dtype, datatype_descriptions.get(dtype, "Description not available")

def overview_description(dtype_df, model):
    dtype_description = []
    dtypes = dtype_df['Data Type'].unique()
    for types in dtypes:
        description = get_datatype_description(types)
        dtype_description.append(description)
    
    dtype_description = pd.DataFrame(dtype_description, columns=['Data Type', 'Description'])

    description_text = "Data Types:\n\n" + "\n\n".join(
        "- " + row['Data Type'] + ": " + row['Description']
        for _, row in dtype_description.iterrows()
    )
    
    # select model
    if model == "Phi3": repo_id = "microsoft/Phi-3-mini-4k-instruct"

    # template 
    template = """
    Define the following data analysis terms in short sentences:

    - Zeros: 

    - Missing Values: 
    
    - Duplicate Rows: 

    Do not include python code or any other code snippets in the answer.

    Answer:
    """
    
    prompt = PromptTemplate(template=template)

    llm = HuggingFaceHub(
        repo_id=repo_id,
        model_kwargs={"temperature": 0.2, "max_new_tokens": 150}
    )

    # LLm Chain object
    llm_chain = LLMChain(prompt=prompt, llm=llm)

    # generate response
    response = llm_chain.invoke({})

    response_text = textwrap.dedent(response.get("text", "")).strip()
    cleaned_text = extract_cleaned_response(response_text)

    # "Answer:"가 없는 경우 처리
    description_text = "Definitions:\n\n" + cleaned_text
    return description_text

# overview explanation
def overview_slm(model):
    # select model
    if model == "Phi3": 
        repo_id = "microsoft/Phi-3-mini-4k-instruct"

    # template 
    template = """
    the dataset has the following overview:
    {overview_table}

    the dataset has the following data types:
    {dtype_table}

    Summary:
    Provide a summary of the overview and data types, excluding sample data where possible.

    Analysis:
    List the point which is crucial to increase the quality of the data analysis process with recommended preprocessing method based on the overview and data types.
    
    Answer:
    """

    # prompt template
    prompt = PromptTemplate(
        template=template,
        input_variables=["overview_table", "dtype_table"]
    )

    # HuggingFaceHub object
    llm = HuggingFaceHub(
        repo_id=repo_id,
        model_kwargs={"temperature": 0.2, "max_new_tokens": 1024}
    )

    # LLm Chain object
    llm_chain = LLMChain(prompt=prompt, llm=llm)

    # generate response
    response = llm_chain.invoke({
        "overview_table": st.session_state.overview_table.to_markdown(),
        "dtype_table": st.session_state.dtype_table.to_markdown()
    })

    response_text = textwrap.dedent(response.get("text", "")).strip()
    cleaned_text = extract_cleaned_response(response_text)

    # "Answer:"가 없는 경우 처리
    description_text = "\n\n" + cleaned_text
    return description_text

def overview_llm(model):
    client = openai.OpenAI(api_key=st.session_state.openai_api_key)

    template = """
    the dataset has the following overview:
    {overview_table}

    the dataset has the following data types:
    {dtype_table}

    Summary:
    Provide a summary of the overview and data types, excluding sample data where possible.

    Analysis:
    List the point which is crucial to increase the quality of the data analysis process with recommended preprocessing method based on the overview and data types.
    
    Answer:
    """

    prompt = template.format(overview_table=st.session_state.overview_table.to_markdown(), dtype_table=st.session_state.dtype_table.to_markdown())
    
    try:
        response = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=2024,
            temperature=0.2
        )
        answer_text = response.choices[0].message.content.strip().split("Answer:")[-1].strip()
    except Exception as e:
        answer_text = f"Error calling GPT API: {e}"

    return answer_text

def overview(df, dtype_df):
    show_overview_table(df, dtype_df)
    show_alerts(df)
    show_dtype_df(dtype_df)

    # duplicate rows
    if df.duplicated().sum() > 0:
        with st.expander("**Duplicate Rows**"):
            st.dataframe(df[df.duplicated()])
    
    # show explanation
    if st.session_state.llm_response:
        if st.session_state.llm_model in ["gpt-3.5-turbo", "gpt-4", "gpt-4o", "gpt-4o-mini"]:
            response = overview_llm(st.session_state.llm_model)
        else:
            description_txt = overview_description(dtype_df, st.session_state.llm_model)
            analyze_txt = overview_slm(st.session_state.llm_model)
            response = description_txt + analyze_txt

        if st.session_state.language != "en":
            response = google_translate(response, "ko")
        
        txt = st.text_area("LLM response", response, height=500)
        st.write(f"Response: {len(txt)} characters.")

##################################### Missing Values ######################################

# missing values msno plot
@st.cache_data
def missing_values_plot(df):
    # Missing Values
    st.markdown("### Missing Values")

    # missing values plots
    with st.container(border=True):
        tab1, tab2, tab3, tab4 = st.tabs(['Matrix', 'Bar', 'Heatmap', 'Dendrogram'])

        # barplot
        with tab1:
            fig, ax = plt.subplots(figsize=(18, 8))
            msno.bar(df, ax=ax)
            st.pyplot(fig)
        
        # matrix
        with tab2:
            fig, ax = plt.subplots(figsize=(18, 8))
            msno.matrix(df, ax=ax)
            st.pyplot(fig)
        
        # heatmap
        with tab3:
            fig, ax = plt.subplots(figsize=(18, 8))
            msno.heatmap(df, ax=ax)
            st.pyplot(fig)

        # dendrogram
        with tab4:
            fig, ax = plt.subplots(figsize=(18, 8))
            msno.dendrogram(df, ax=ax)
            st.pyplot(fig)

# missing values description
def missing_values_description(model):
    # select model
    if model == "Phi3": 
        repo_id = "microsoft/Phi-3-mini-4k-instruct"

    # template 
    template = """
    Define the following data analysis terms in short sentences:

    - Missing Values: 

    Define the following visualization methods to analyze missing values in short sentences:

    - Matrix: 
    - Bar plot:
    - Heatmap: 
    - Dendrogram:
    
    Do not include python code or any other code snippets in the answer.

    Answer:
    """

    # prompt template
    prompt = PromptTemplate(
        template=template,
    )

    # HuggingFaceHub object
    llm = HuggingFaceHub(
        repo_id=repo_id,
        model_kwargs={"temperature": 0.2, 
                    "max_new_tokens" : 250}
    )

     # LLm Chain object
    llm_chain = LLMChain(prompt=prompt, llm=llm)

    # generate response
    response = llm_chain.invoke({})

    response_text = textwrap.dedent(response.get("text", "")).strip()
    cleaned_text = extract_cleaned_response(response_text)

    # "Answer:"가 없는 경우 처리
    description_text = "About Missing Values\n\n" + cleaned_text
    return description_text

# missing values explanation
def missing_values_slm(model):
    # select model
    if model == "Phi3": 
        repo_id = "microsoft/Phi-3-mini-4k-instruct"

   # template 
    template = """
    Here is a summary of the Missing Values in the dataset: {missing_table}
    The table consists of columns name and the number of missing values in each column.
    The total rows of the dataset are {rows}.

    Definition of Missing Values:  
    Missing values refer to the absence of data in one or more columns of a dataset. This can occur due to various reasons, such as data entry errors, incomplete data collection, or limitations in data integration from multiple sources.
    
    About Missing Values:
    a. Impact of Missing Values on the Dataset
    Provide a list of columns with more than one missing value, along with an explanation of why these missing values could be problematic for analysis.
    b. Recommendation for Handling Missing Values 
    Provide recommendations for handling missing values in each column with more than one missing value, along with reasons why the suggested method is appropriate.  
    
    Format of answer: 
    [Column Name]
    - Reason why the missing values could be problematic
    - Recommendation
    Example: 
    "Age"
    - This column is crucial for prediction and has over 50% missing data. Missing values in age can lead to biased estimates and affect the accuracy of the analysis."
    - Impute missing values with the median - Median imputation is a common method for handling missing values in continuous variables."

    Definition of the Methods:
    Explain the methods to handle missing values which are recommended in the previous step.
    Example: 
    - Imputation: "Imputation is the process of replacing missing values with substituted values. Common imputation methods include mean, median, mode imputation, and K-nearest neighbors (KNN) imputation."

    Answer:
    """

    # generate prompt template 
    prompt = PromptTemplate(
        template=template,
        input_variables=["missing_table", "rows"]
    )

    # HuggingFaceHub object
    llm = HuggingFaceHub(
        repo_id=repo_id,
        model_kwargs={"temperature": 0.2, 
                    "max_new_tokens" : 1024}
    )

    # LLm Chain object
    llm_chain = LLMChain(prompt=prompt, llm=llm)

    # generate response
    response = llm_chain.invoke({
        "missing_table": st.session_state.missing_table.to_markdown(),
        "rows": st.session_state.df.shape[0]
    })

    response_text = textwrap.dedent(response.get("text", "")).strip()
    cleaned_text = extract_cleaned_response(response_text)

    # "Answer:"가 없는 경우 처리
    description_text = "\n\nMissing Values Summary: \n\n" + cleaned_text
    return description_text

def missing_values_llm(model):
    client = openai.OpenAI(api_key=st.session_state.openai_api_key)

    template = """
    Here is a summary of the Missing Values in the dataset: {missing_table}
    The table consists of columns name and the number of missing values in each column.
    The total rows of the dataset are {rows}.

    Definition of Missing Values:  
    Missing values refer to the absence of data in one or more columns of a dataset. This can occur due to various reasons, such as data entry errors, incomplete data collection, or limitations in data integration from multiple sources.
    
    About Missing Values:
    a. Impact of Missing Values on the Dataset
    Provide a list of columns with more than one missing value, along with an explanation of why these missing values could be problematic for analysis.
    b. Recommendation for Handling Missing Values 
    Provide recommendations for handling missing values in each column with more than one missing value, along with reasons why the suggested method is appropriate.  
    
    Format of answer: 
    [Column Name]
    - Reason why the missing values could be problematic
    - Recommendation
    Example: 
    "Age"
    - This column is crucial for prediction and has over 50% missing data. Missing values in age can lead to biased estimates and affect the accuracy of the analysis."
    - Impute missing values with the median - Median imputation is a common method for handling missing values in continuous variables."

    Definition of the Methods:
    Explain the methods to handle missing values which are recommended in the previous step.
    Example: 
    - Imputation: "Imputation is the process of replacing missing values with substituted values. Common imputation methods include mean, median, mode imputation, and K-nearest neighbors (KNN) imputation."

    Answer:
    """

    prompt = template.format(missing_table=st.session_state.missing_table.to_markdown(), rows=st.session_state.df.shape[0])
    try:
        response = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=2024,
            temperature=0.2
        )
        answer_text = response.choices[0].message.content.strip().split("Answer:")[-1].strip()
    except Exception as e:
        answer_text = f"Error calling GPT API: {e}"

    return answer_text

# missing values page
def missing_values(df):
    # missing values plot
    missing_values_plot(df)

    # missing values table
    with st.expander("**Rows with Missing Values**"):
        missing_values_df = df[df.isna().any(axis=1)]
        st.dataframe(missing_values_df)
    
    missing_table = pd.DataFrame(df.isna().sum())
    missing_table.columns = ['Missing Values']
    st.session_state.missing_table = missing_table

    # missing values explanation
    if st.session_state.llm_response:
        if st.session_state.llm_model in ["gpt-3.5-turbo", "gpt-4", "gpt-4o", "gpt-4o-mini"]:
            response = missing_values_llm(st.session_state.llm_model)
        else:
            description_txt = missing_values_description(st.session_state.llm_model)
            analyze_txt = missing_values_slm(st.session_state.llm_model)
            response = description_txt + analyze_txt

        if st.session_state.language != "en":
            response = google_translate(response, "ko")
        
        txt = st.text_area("LLM response", response, height=500)
        st.write(f"Response: {len(txt)} characters.")
##################################### Outlier ######################################

# detect outliers by z-score
def detect_outliers_zscore(df, column, threshold=3):
    mean = df[column].mean()
    std = df[column].std()
    zscore = (df[column] - mean) / std
    outliers = df[np.abs(zscore) > threshold]
    return outliers, outliers.shape[0]

# show zscore
def show_zscore(df, numeric_col, detail):
    z_outliers = {}
    z_outlier_df = {}

    for col in numeric_col:
        z_outlier_df[col], z_outliers[col] = detect_outliers_zscore(df, col)
    z_score_df = pd.DataFrame(z_outliers, index=['Outliers']).T
    st.dataframe(z_score_df)

    if detail:
        for col in numeric_col:
            if not z_outlier_df[col].empty:
                with st.expander(f"**Outliers in {col} (Count: {z_outliers[col]})**"):
                    st.dataframe(z_outlier_df[col])
    
    return z_score_df

# detect outliers by IQR
def detect_outliers_IQR(df, column):
    Q1 = df[column].quantile(0.25)
    Q2 = df[column].quantile(0.50)
    Q3 = df[column].quantile(0.75)
    min_val = df[column].min()
    max_val = df[column].max()

    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR

    outliers = df[(df[column] < lower_bound) | (df[column] > upper_bound)]
    non_outlier = df[(df[column] >= lower_bound) & (df[column] <= upper_bound)]
    
    result = {
        "Outliers": f'{outliers.shape[0]}',
        "Non-Outliers": f'{non_outlier.shape[0]}',
        "min": f'{min_val:.2f}',
        "max": f'{max_val:.2f}',
        "Q1": f'{Q1:.2f}',
        "Q2": f'{Q2:.2f}', 
        "Q3": f'{Q3:.2f}',
        "IQR": f'{IQR:.2f}',
        "Lower Bound": f'{lower_bound:.2f}',
        "Upper Bound": f'{upper_bound:.2f}',
    }

    return outliers, result

# show iqr
def show_IQR(df, numeric_col, detail):
    iqr_outliers = {}
    iqr_outlier_df = {}

    for col in numeric_col:
        iqr_outlier_df[col], iqr_outliers[col] = detect_outliers_IQR(df, col)
    iqr_df = pd.DataFrame(iqr_outliers).T
    st.dataframe(iqr_df)

    if detail:
        for col in numeric_col:
            if not iqr_outlier_df[col].empty:
                with st.expander(f"**Outliers in {col} (Count: {iqr_outliers[col]['Outliers']})**"):
                    st.dataframe(iqr_outlier_df[col])
    
    return iqr_df

# show outlier
def show_outliers(df, numeric_col):
    st.markdown("### Outlier")

    # detail info checkbox
    detail = st.checkbox("Show Details", value=False)

    # z-score
    with st.container(border=True):
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("**Z-Score**")
            z_score_df = show_zscore(df, numeric_col, detail)
        
        with col2:
            st.markdown("**IQR**")
            iqr_df = show_IQR(df, numeric_col, detail)
            
        # Combine Z-score and IQR outlier detection results
        outlier_df = pd.concat([z_score_df, iqr_df['Outliers']], axis=1).astype(int)
        outlier_df.columns = ['Z-Score', 'IQR']

        # Filter out rows where both Z-Score and IQR have 0 outliers
        outlier_df = outlier_df[(outlier_df['Z-Score'] != 0) | (outlier_df['IQR'] != 0)]

        # Save the outlier dataframe if there are outliers
        if not outlier_df.empty:
            st.session_state.outlier_df = outlier_df
        else:
            st.session_state.outlier_df = pd.DataFrame()

# boxplot
@st.cache_data
def draw_boxplot(df, column):
    fig = plt.figure(figsize = (18, 8))
    sns.boxplot(x=df[column])
    return fig

# outlier description
def outlier_description(model):
    # model
    if model == "Phi3":
        repo_id = "microsoft/Phi-3-mini-4k-instruct"

    # template 
    template = """
    Define the following data analysis terms in short sentences:

    - Outliers: 

    Z-score and IQR are two common methods to detect outliers in a dataset.
    Define the following terms and Provide the range of values that are considered outliers for each method in the short sentences:
        
    - Z-score: 

    - IQR (Interquartile Range):

    Answer:
    """

    # prompt template
    prompt = PromptTemplate(
        template=template,
    )

    # HuggingFaceHub object
    llm = HuggingFaceHub(
        repo_id=repo_id,
        model_kwargs={"temperature": 0.2, 
                    "max_new_tokens" : 300}
    )

    # LLm Chain object
    llm_chain = LLMChain(prompt=prompt, llm=llm)

    # generate response
    response = llm_chain.invoke({})

    response_text = textwrap.dedent(response.get("text", "")).strip()
    cleaned_text = extract_cleaned_response(response_text)

    # "Answer:"가 없는 경우 처리
    description_text = "About Outlier Detection: \n\n" + cleaned_text
    return description_text

def outlier_slm(model):
    # model
    if model == "Phi3": 
        repo_id = "microsoft/Phi-3-mini-4k-instruct"

    # template
    template = """
    Outlier table: {outlier_table}
    The outlier table consists of columns' names and the number of outliers detected using Z-Score and IQR methods.
    The total rows of the dataset are {rows}.
    Columns having more than one outliers: {col}

    About outliers:
    Provide the Number of outliers and the exptected effect of outlier of {col} and Recommend method to handle outliers of each columns.
    Follow the format below:
    "Column Name"
    - Reason why the outliers could be problematic
    - Recommended method to handle outliers
    Here is an Example:
    "Age"
    - This column has a significant number of outliers. Outliers in age can lead to biased estimates and affect the accuracy of the analysis.
    - Use Winsorization to handle outliers - Winsorization is a method that replaces extreme values with less extreme values to reduce the impact of outliers on the analysis.

    Answer: 
    """

    # generate prompt template
    prompt = PromptTemplate(
        template=template,
        input_variables=["outlier_table", "rows", "col"]
    )

    # HuggingFaceHub object
    llm = HuggingFaceHub(
        repo_id=repo_id,
        model_kwargs={"temperature": 0.2, 
                      "max_new_tokens": 1024}
    )

    # LLm Chain object
    llm_chain = LLMChain(prompt=prompt, llm=llm)

    # Generate response
    response = llm_chain.invoke({
        "outlier_table": st.session_state.outlier_df.to_markdown(),
        "rows": st.session_state.df.shape[0],
        "col": st.session_state.outlier_df.index.to_list()
    })

    response_text = textwrap.dedent(response.get("text", "")).strip()
    cleaned_text = extract_cleaned_response(response_text)

    # "Answer:"가 없는 경우 처리
    description_text = "\n\nAbout Outlier Detection: \n\n" + cleaned_text
    return description_text

def outlier_llm(model, timeout=30):
    client = openai.OpenAI(api_key=st.session_state.openai_api_key)

    template = """
    Outlier table: {outlier_table}
    The outlier table consists of columns' names and the number of outliers detected using Z-Score and IQR methods.
    The total rows of the dataset are {rows}.
    Columns having more than one outliers: {col}

    About outliers:
    Provide the Number of outliers and the exptected effect of outlier of {col} and Recommend method to handle outliers of each columns.
    Follow the format below:
    "Column Name"
    - Reason why the outliers could be problematic
    - Recommended method to handle outliers
    Here is an Example:
    "Age"
    - This column has a significant number of outliers. Outliers in age can lead to biased estimates and affect the accuracy of the analysis.
    - Use Winsorization to handle outliers - Winsorization is a method that replaces extreme values with less extreme values to reduce the impact of outliers on the analysis.

    Answer: 
    """

    prompt = template.format(outlier_table=st.session_state.outlier_df.to_markdown(), rows=st.session_state.df.shape[0], col=st.session_state.outlier_df.index.to_list())

    try:
        response = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=2024,
            temperature=0.2
        )
        answer_text = response.choices[0].message.content.strip().split("Answer:")[-1].strip()
    except Exception as e:
        answer_text = f"Error calling GPT API: {e}"

    return answer_text

# outlier
def outlier(df, dtype_df):
    numeric_col = dtype_df[dtype_df['Data Type'].isin(['Numeric (Discrete)', 'Numeric (Continuous)'])].index.to_list()

    # show outliers
    show_outliers(df, numeric_col)

    # boxplot
    with st.container(border=True):
        st.markdown("**Boxplot**")
        option = st.selectbox(
            "Select the column that you want to draw boxplot",
            numeric_col
        )
        boxplot_fig = draw_boxplot(df, option)
        st.pyplot(boxplot_fig)
    
    # outlier explanation
    if st.session_state.llm_response:
        if st.session_state.llm_model in ["gpt-3.5-turbo", "gpt-4", "gpt-4o", "gpt-4o-mini"]:
            response = outlier_llm(st.session_state.llm_model)
        else:
            description_txt = outlier_description(st.session_state.llm_model)
            analyze_txt = outlier_slm(st.session_state.llm_model)
            response = description_txt + analyze_txt

        if st.session_state.language != "en":
            response = google_translate(response, "ko")
        
        txt = st.text_area("LLM response", response, height=500)
        st.write(f"Response: {len(txt)} characters.")

