"""
This is the Visualization page

"""

###################################### import ######################################

# library
import streamlit as st
import pandas as pd
import numpy as np
import gettext
import os

from st_mui_table import st_mui_table
import matplotlib.pyplot as plt
import plotly.express as px

from statsmodels.graphics.mosaicplot import mosaic
from statsmodels.tsa.seasonal import seasonal_decompose
from pmdarima import auto_arima
from sklearn.model_selection import train_test_split

import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from collections import Counter
from konlpy.tag import Okt

import re
from langchain import LLMChain
from langchain.prompts import PromptTemplate
from langchain.llms import HuggingFaceHub
from deep_translator import GoogleTranslator
import textwrap
import openai

###################################### set ##############

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

###################################### Categorical ######################################

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
        showIndex=False,
        key=key
    )

# categorical info
@st.cache_data
def categorical_info(df, col):
    distinct = df[col].nunique()
    distinct_per = df[col].nunique() / len(df[col]) * 100
    missing = df[col].isna().sum()
    missing_per = df[col].isna().sum() / len(df[col]) * 100
    memory_size = df[col].memory_usage(deep=True) / 1024 ** 2
    value_count = df[col].value_counts()

    # result
    result = pd.DataFrame({
        'Metrics': ['Distinct', 'Distinct (%)', 'Missing', 'Missing (%)', 'Memory Size (MB)'],
        'Values': [
            f'{distinct}',
            f'{distinct_per:.2f}%',
            f'{missing}',
            f'{missing_per:.2f}%',
            f'{memory_size:.2f} MB',
        ]
    })

    # value count
    missing_df = pd.DataFrame({'Values': ['Missing'], 'Counts': [missing]})
    value_count_df = pd.DataFrame({
        'Values': value_count.index,
        'Counts': value_count.values
    })    
    value_count_df = pd.concat([value_count_df, missing_df], ignore_index=True)

    return result, value_count_df

# Set the desired width and height as a percentage of the default size
width = 800 * 0.8  # 80% of the default width
height = 400 * 0.8  # 80% of the default height

# categorical barplot
@st.cache_data
def categorical_barplot(df, col):
    value_count = df[col].value_counts()
    
    # Create interactive bar plot using plotly
    fig = px.bar(
        value_count,
        x=value_count.index,
        y=value_count.values,
        labels={'x': 'Category', 'y': 'Count'},
        color=value_count.index,
        color_discrete_sequence=px.colors.qualitative.Pastel 
    )

    # Customize hover information
    fig.update_traces(hovertemplate='%{x}: %{y}')

    # Remove title and legend
    fig.update_layout(title='', showlegend=False)

    # Set the size of the figure
    fig.update_layout(width=width, height=height)

    return fig

# categorical pie chart
@st.cache_data
def categorical_pie(df, col):
    value_count = df[col].value_counts(normalize=True) * 100

    # Create interactive pie chart using plotly
    fig = px.pie(
        values=value_count, 
        names=value_count.index, 
        color_discrete_sequence=px.colors.qualitative.Pastel
    )

    # Customize hover information
    fig.update_traces(hovertemplate='%{label}: %{value:.1f}%')

    # Remove title and legend
    fig.update_layout(title='', showlegend=False)

    # Set the size of the figure
    fig.update_layout(width=width, height=height)

    return fig

# categorical slm
def analyze_column(df, column_name):
    """
    Analyzes a specific column in a DataFrame by calculating technical statistics
    and value counts.

    Parameters:
    df (pd.DataFrame): The input DataFrame.
    column_name (str): The column to analyze.

    Returns:
    stats_df (pd.DataFrame): DataFrame containing the calculated statistics.
    value_count_df (pd.DataFrame): DataFrame containing value counts for the column.
    stats_table_str (str): String representation of the statistics table.
    value_counts_table_str (str): String representation of the value counts table.
    """
    # Calculate technical statistics
    distinct = df[column_name].nunique()
    missing = df[column_name].isna().sum()
    missing_per = missing / len(df[column_name]) * 100
    memory_size = df[column_name].memory_usage(deep=True)

    # Create the value counts DataFrame
    value_count_df = df[column_name].value_counts().reset_index()
    value_count_df.columns = [f"{column_name.capitalize()} Value", "Count"]

    # Convert tables to strings for easier display
    value_counts_table_str = value_count_df.to_string(index=False)
    return distinct, missing, missing_per, memory_size, value_counts_table_str

def categorical_slm(df, column_name, categorical, model):
    # select model
    if model == "Phi3": 
        repo_id = "microsoft/Phi-3-mini-4k-instruct"

    # template 
    template = """
    Here is an analysis of the '{column_name}' feature in the dataset, based on definitions, characteristics of the column, and preprocessing considerations.

    Term Definitions:
    - Distinct Values: The count of unique values in the feature.
    - Missing Values: Indicates the absence of data values in the feature.
    - Memory Usage: The amount of memory required to store this feature.

    Analysis of '{column_name}' Feature:
    Summarize the key characteristics of the '{column_name}' feature in one or two sentences base on the provided statistics.
    - Distinct Value: {distinct_count}
    - Missing Value: {missing_count} ({missing_percentage}%)
    - Memory Usage: {memory_size} bytes
    - Value Distribution: {value_counts_table}

    Preprocessing Tips:
    Provide maximum 3 preprocessing tips based on the analysis of the '{column_name}' feature.

    Answer maxiumum 500 characters.
    Answer:
    """

    # prompt template
    prompt = PromptTemplate(
        template=template,
        input_variables=["column_name", "column_type", "value_counts_table", "distinct_count", "missing_count", "missing_percentage", "memory_size"]
    )

    # HuggingFaceHub object
    llm = HuggingFaceHub(
        repo_id=repo_id,
        model_kwargs={"temperature": 0.2, "max_new_tokens": 512},
    )

    # LLm Chain object
    llm_chain = LLMChain(prompt=prompt, llm=llm)

    # data analysis
    if categorical:
        column_type = "Categorical"
    else:
        column_type = "Boolean"
    
    # analyze column
    distinct, missing, missing_per, memory_size, value_counts_table_str = analyze_column(df, column_name)
    
    # generate response
    response = llm_chain.invoke({
        "column_name": column_name,
        "column_type": column_type,
        "value_counts_table": value_counts_table_str,
        "distinct_count": distinct,
        "missing_count": missing,
        "missing_percentage": round(missing_per, 2),
        "memory_size": memory_size
    })

    response_text = textwrap.dedent(response.get("text", "")).strip()
    cleaned_text = extract_cleaned_response(response_text)

    fixed_intro = f"Here is an analysis of the '{column_name}' feature in the dataset, based on definitions, characteristics of the column, and preprocessing considerations."

    # "Answer:"가 없는 경우 처리
    description_text = fixed_intro + "\n" + cleaned_text
    return description_text

# categorical llm
def categorical_llm(df, column_name, categorical, model):
    client = openai.OpenAI(api_key=st.session_state.openai_api_key)

    template = """
    The following is an analysis of the '{column_name}' functionality of the dataset based on definitions, column properties, and preprocessing considerations.

    Content analysis of '{column_name}' feature: '{column_name}' column records information associated with each item in the dataset and is of type '{column_type}'.
    - Unique value: This column contains {distinct_count} unique values.
    - Missing value: There are {missing_count} missing values, accounting for {missing_percentage}% of the data.
    - Memory Usage: This column uses approximately {memory_size} bytes of memory.
    - Value Distribution:
    {value_counts_table}

    Explain the content analysis of the '{column_name}' feature and recommend processing and preventive measures based on this.
    Answer maxiumum 500 characters.
    Answer:
    """

    # data analysis
    if categorical:
        column_type = "Categorical"
    else:
        column_type = "Boolean"

    # analyze column
    distinct, missing, missing_per, memory_size, value_counts_table_str = analyze_column(df, column_name)

    prompt = template.format(column_name=column_name, column_type=column_type, distinct_count=distinct,
                         missing_count=missing, missing_percentage=missing_per, memory_size=memory_size,
                         value_counts_table=value_counts_table_str)
 
    response = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        max_tokens=2024,
        temperature=0.2
    )

    answer_text = response.choices[0].message.content.strip().split("Answer:")[-1].strip()
    return answer_text

def categorical_format(df, categorical):
    for i, col in enumerate(df.columns):
        with st.expander(f"**{col}**", expanded=True):
            col1, col2 = st.columns(2)
            col3, col4 = st.columns(2)
                            
            # Use cached functions for the plots and data
            info, value_count = categorical_info(df, col)
            info_add = pd.DataFrame({'Metrics': [''], 'Values': ['']})
            info_table = pd.concat([info, info_add], ignore_index=True)

            # barplot
            with col1:
                fig1 = categorical_barplot(df, col)
                st.plotly_chart(fig1, key=f'barplot_{col}_{i}')
                            
            # pie chart
            with col2:
                fig2 = categorical_pie(df, col)
                st.plotly_chart(fig2, key=f'piechart_{col}_{i}')

            # info table
            with col3:
                if categorical:
                    show_table(info_table, key=f"Info_categorical_{i}", n=6)
                else:
                    show_table(info_table, key=f"Info_bool_{i}", n=5)
                            
            # value count table
            with col4:
                if categorical:
                    show_table(value_count, key=f"Value_count_categorical_{i}", n=value_count.shape[0]+1)
                else:
                    show_table(value_count, key=f"Value_count_bool_{i}", n=value_count.shape[0])

            # show explanation
            if st.session_state.llm_response:
                if st.session_state.llm_model in ["gpt-3.5-turbo", "gpt-4", "gpt-4o", "gpt-4o-mini"]:
                    response = categorical_llm(df, col, categorical, st.session_state.llm_model)
                else:
                    response = categorical_slm(df, col, categorical, st.session_state.llm_model)

                if st.session_state.language != "en":
                    response = google_translate(response, "ko")

                txt = st.text_area("LLM response", response, height=300)
                st.write(f"Response: {len(txt)} characters.")

def categorical_description():
    fixed_intro = f"""Term Definitions:
    - Distinct Values: The count of unique values in the feature.
    - Missing Values: Indicates the absence of data values in the feature.
    - Memory Usage: The amount of memory required to store this feature.
    """

    return fixed_intro

# Categorical page
def categorical(df_cat, df_bool):
    # title
    st.markdown("### Categorical")

    with st.container(border=True):
        tab1, tab2 = st.tabs(['Categorical', 'Boolean'])

        # categorical
        with tab1:
            if df_cat.shape[1] > 0:        
                categorical_format(df_cat, True)
            else:
                st.warning(_("No Categorical Columns"))

        # boolean
        with tab2:
            if df_bool.shape[1] > 0:
                categorical_format(df_bool, False)
            else:
                st.warning(_("No Boolean Columns"))

    description = categorical_description()        

    if st.session_state.language != "en":
        description = google_translate(description, "ko")

    txt = st.text_area("LLM response", description, height=150, key="categorical_description")
    st.write(f"Response: {len(txt)} characters.")


###################################### Numerical ######################################

# numerical info
def numerical_info(df, col):
    distinct = df[col].nunique()
    distinct_per = distinct / len(df) * 100
    missing = df[col].isnull().sum()
    missing_per = missing / len(df) * 100
    infinit = df[col].isin([np.inf, -np.inf]).sum()
    infinit_per = infinit / len(df) * 100
    memory_size = df[col].memory_usage(deep=True) / 1024 ** 2

    result = pd.DataFrame({
        'Metrics': ['Distinct', 'Distinct (%)', 'Missing', 'Missing (%)', 'Infinit', 'Infinit (%)', 'Memory Size (MB)'],
        'Values': [
            f'{distinct}',
            f'{distinct_per:.2f}%',
            f'{missing}',
            f'{missing_per:.2f}%',
            f'{infinit}',
            f'{infinit_per:.2f}%',
            f'{memory_size:.2f} MB',
        ]
    })

    description = df[col].describe().to_list()
    description_df = pd.DataFrame({
        'Metrics': ['Count', 'Mean', 'Std', 'Min', '25%', '50%', '75%', 'Max'],
        'Values': description
    })
    nan_df = pd.DataFrame({'Metrics': ['NaN'], 'Values': [missing]})
    description_df = pd.concat([description_df, nan_df], ignore_index=True)
    
    return result, description_df

# Set the desired width and height as a percentage of the default size
width = 800 * 0.8  # 80% of the default width
height = 400 * 0.8  # 80% of the default height

# discrete plot
@st.cache_data
def discrete_plot(df, col):
    value_counts = df[col].value_counts().sort_values(ascending=False)
    
    # Create a bar plot using plotly
    fig = px.bar(
        x=value_counts.index,
        y=value_counts.values,
        labels={'x': 'Category', 'y': 'Count'},
        color=value_counts.index,
        color_discrete_sequence=px.colors.qualitative.Pastel 
    )

    # Remove title and legend
    fig.update_layout(title='', showlegend=False)

    # Customize the color bar (note that it may still be auto-generated based on color)
    fig.update_traces(marker=dict(color='skyblue'))

    # Set the size of the figure
    fig.update_layout(width=width, height=height)

    return fig


# continuous plot
@st.cache_data
def continuous_plot(df, col):
    # Create a histogram with KDE using plotly
    fig = px.histogram(
        df,
        x=col,
        color_discrete_sequence=['skyblue'],
        marginal='rug',
        histnorm='probability density'
    )

    # Remove title and legend
    fig.update_layout(title='', showlegend=False)

    # Set the size of the figure
    fig.update_layout(width=width, height=height)

    return fig

# boxplot
@st.cache_data
def boxplot(df, col):
    # Create a box plot using plotly
    fig = px.box(
        df,
        x=col,
        color_discrete_sequence=['skyblue']
    )

    # Remove title and legend
    fig.update_layout(title='', showlegend=False)

    # Set the size of the figure
    fig.update_layout(width=width, height=height)

    return fig

# numerical slm
def numeric_preprocessing(series):
    distinct_count = series.nunique()
    missing_count = series.isnull().sum()
    missing_percentage = (missing_count / len(series)) * 100
    mean = series.mean()
    minimum = series.min()
    maximum = series.max()
    variance = series.var()
    skewness = series.skew()
    kurtosis = series.kurtosis()

    result = {
        "distinct_count": distinct_count,
        "missing_count": missing_count,
        "missing_percentage": round(missing_percentage, 2),
        "mean": mean,
        "minimum": minimum,
        "maximum": maximum,
        "variance": variance,
        "skewness": skewness,
        "kurtosis": kurtosis,
    }
    return result

def numerical_slm(df, column_name, model):
    # select model
    if model == "Phi3": 
        repo_id = "microsoft/Phi-3-mini-4k-instruct"

    # template 
    template = """
    Here is a summary of the statistics for the numeric feature '{column_name}' in the dataset:

    Term Definitions:
    - Distinct Values: The count of unique values in the feature.
    - Missing Values: Indicates the absence of data values in the feature.
    - Skewness: Degree of symmetry of the distribution.
    - Kurtosis: Degree of sharpness of the center of the distribution.

    Analysis of '{column_name}' Feature:
    Summarize the key characteristics of the '{column_name}' feature in one or two sentences based on the provided statistics.
    - Mean: {mean}
    - Minimum: {minimum}
    - Maximum: {maximum}
    - Distinct Values: {distinct_count}
    - Missing Values: {missing_count} ({missing_percentage}%)
    - Variance: {variance}
    - Skewness: {skewness}
    - Kurtosis: {kurtosis}

    Preprocessing Tips:
    Provide maximum 3 preprocessing tips based on the analysis of the '{column_name}' feature.

    Answer maxiumum 500 characters.
    Answer:
    """

    # prompt template
    prompt = PromptTemplate(
        template=template,
        input_variables=[
            "column_name", "mean", "minimum", "maximum", "distinct_count",
            "missing_count", "missing_percentage", "variance",
            "skewness", "kurtosis"
        ]
    )

    # HuggingFaceHub object
    llm = HuggingFaceHub(
        repo_id=repo_id,
        model_kwargs={"temperature": 0.2, "max_new_tokens": 512},
    )

    # LLm Chain object
    llm_chain = LLMChain(prompt=prompt, llm=llm)

    # data analysis
    stats = numeric_preprocessing(df[column_name])
    
    # generate response
    response = llm_chain.invoke({
        "column_name": column_name,
        "mean": stats["mean"],
        "minimum": stats["minimum"],
        "maximum": stats["maximum"],
        "distinct_count": stats["distinct_count"],
        "missing_count": stats["missing_count"],
        "missing_percentage": stats["missing_percentage"],
        "variance": stats["variance"],
        "skewness": stats["skewness"],
        "kurtosis": stats["kurtosis"]
    })

    response_text = textwrap.dedent(response.get("text", "")).strip()
    cleaned_text = extract_cleaned_response(response_text)

    fixed_intro = f"Here is an analysis of the '{column_name}' feature in the dataset, based on definitions, characteristics of the column, and preprocessing considerations."

    # "Answer:"가 없는 경우 처리
    description_text = fixed_intro + "\n" + cleaned_text
    return description_text

# numerical llm
def numerical_llm(df, column_name, model):
    client = openai.OpenAI(api_key=st.session_state.openai_api_key)

    template = """
    Here is a summary of the statistics for the numeric feature '{column_name}' in the dataset:

    Basic Statistics:
    - Mean: {mean}
    - Minimum: {minimum}
    - Maximum: {maximum}
    - Distinct Values: {distinct_count}
    - Missing Values: {missing_count} ({missing_percentage}%)
    - Variance: {variance}
    - Skewness: {skewness}
    - Kurtosis: {kurtosis}

    Explain the content analysis of the '{column_name}' feature and recommend processing and preventive measures based on the above Basic Statistics.
    Answer maxiumum 500 characters.

    Answer:
    """

    # data analysis
    stats = numeric_preprocessing(df[column_name])
    
    prompt = template.format(
        column_name=column_name,
        mean=stats["mean"],
        minimum=stats["minimum"],
        maximum=stats["maximum"],
        distinct_count=stats["distinct_count"],
        missing_count=stats["missing_count"],
        missing_percentage=stats["missing_percentage"],
        variance=stats["variance"],
        skewness=stats["skewness"],
        kurtosis=stats["kurtosis"],
    )
 
    if column_name is not None:
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
    else:
        answer_text = "No numeric columns found, skipping OpenAI API call."

    return answer_text

def numerical_format(df, discrete):
    for col in df.columns:
        with st.expander(f"**{col}**", expanded=True):
            col1, col2 = st.columns(2)
            col3, col4 = st.columns(2)
            info, description = numerical_info(df, col)
            info_add = pd.DataFrame({'Metrics': [''], 'Values': ['']})
            info_table = pd.concat([info, info_add], ignore_index=True)

            with col1:
                if discrete:
                    fig1 = discrete_plot(df, col)
                else:
                    fig1 = continuous_plot(df, col)
                st.plotly_chart(fig1, key=f'discrete_continuous_{col}')
                        
            with col2:
                fig2 = boxplot(df, col)
                st.plotly_chart(fig2, key=f'boxplot_{col}') 
                        
            with col3:
                if discrete:
                    show_table(info_table, key=f"Info_Discrete_{col}", n=8)
                else:
                    show_table(info_table, key=f"Info_Continuous_{col}", n=8)
                        
            with col4:
                if discrete:
                    show_table(description, key=f"Description_Discrete_{col}", n=9)
                else:
                    show_table(description, key=f"Description_Continuous_{col}", n=9)
            
            # show explanation
            if st.session_state.llm_response:
                if st.session_state.llm_model in ["gpt-3.5-turbo", "gpt-4", "gpt-4o", "gpt-4o-mini"]:
                    response = numerical_llm(df, col, st.session_state.llm_model)
                else:
                    response = numerical_slm(df, col, st.session_state.llm_model)

                if st.session_state.language != "en":
                    response = google_translate(response, "ko")

                txt = st.text_area("LLM response", response, height=200)
                st.write(f"Response: {len(txt)} characters.")
 
def numerical_description():
    fixed_intro = f"""Term Definitions:
    - Distinct Values: The count of unique values in the feature.
    - Missing Values: Indicates the absence of data values in the feature.
    - Memory Usage: The amount of memory required to store this feature.
    """

    return fixed_intro

# Numerical page
def numerical(discrete, continuous):
    # title
    st.markdown("### Numerical")

    with st.container(border=True):
        tab1, tab2 = st.tabs(['Discrete', 'Continuous'])

        with tab1:
            if discrete.shape[1] > 0:
                numerical_format(discrete, True)

            else:
                st.warning(_("No Discrete Columns"))

        with tab2:
            if continuous.shape[1] > 0:
                numerical_format(continuous, False)

            else:
                st.warning(_("No Continuous Columns"))
    
    description = numerical_description()

    if st.session_state.language != "en":
        description = google_translate(description, "ko")

    txt = st.text_area("LLM response", description, height=150, key="numerical_description")
    st.write(f"Response: {len(txt)} characters.")

###################################### TimeSeries ######################################

# timeseries plot
@st.cache_data
def timeseries_plot(df, col):
    decomposition = seasonal_decompose(df[col], model='additive', period=1)
    fig, (ax1, ax2, ax3, ax4) = plt.subplots(4, 1, figsize=(15, 8))
    decomposition.observed.plot(ax=ax1)
    ax1.set_title('Observed')
    decomposition.trend.plot(ax=ax2)
    ax2.set_title('Trend')
    decomposition.seasonal.plot(ax=ax3)
    ax3.set_title('Seasonal')
    decomposition.resid.plot(ax=ax4)
    ax4.set_title('Residual')
    return fig

# arima plot
@st.cache_data
def arima_plot(df, col):
    train, test = train_test_split(df[col], test_size=0.2, shuffle=False)
    model = auto_arima(train, seasonal=False, m=12, max_p=2, max_q=2, max_P=1, max_Q=1,
                       stepwise=True, trace=False, n_jobs=-1, maxiter=50)
    model.fit(train)

    forecast = model.predict(n_periods=len(test))
    test = test.to_frame()
    test['forecast'] = forecast

    fig, ax = plt.subplots(figsize=(15, 5))
    ax.plot(train.index, train, label='Train')
    ax.plot(test.index, test[col], label='Test')
    ax.plot(test.index, test['forecast'], label='Forecast')
    ax.legend()

    st.session_state.model_summary = model.summary().as_text()

    return fig, model.summary()

# timeseries page
def display_timeseries(df, col):
    with st.expander(f"**{col}**"):
        fig1 = timeseries_plot(df, col)
        st.pyplot(fig1)
        if col == st.session_state.target:
            fig2, summary = arima_plot(df, col)
            st.pyplot(fig2)
            st.write(summary)

# timeseries slm
def timeseries_slm(df, target_col, model):
    # select model
    if model == "Phi3": 
        repo_id = "microsoft/Phi-3-mini-4k-instruct"

    # template 
    template = """
    The Target data of the dataset contains timeseries data with the following columns:
    {column_names}

    The target column '{target_col}' has been analyzed with the following steps:

    1. **Auto ARIMA Model Summary**:
    - Below is the summary of the ARIMA model:
        ```
        {model_summary}
        ```
    - Based on this summary, provide an interpretation of the ARIMA model components and any key diagnostic statistics.

    2. **Forecasting Results**:
    - The ARIMA model predicted future values of the test set. Based on the model summary, we describe the prediction results and mention considerations.
    Answer:
    """

    # prompt template
    prompt = PromptTemplate(
        template=template,
        input_variables=["column_names", "target_col", "model_summary"]
    )

    # HuggingFaceHub object
    llm = HuggingFaceHub(
        repo_id=repo_id,
        model_kwargs={"temperature": 0.2, "max_new_tokens": 1024},
    )

    # LLm Chain object
    llm_chain = LLMChain(prompt=prompt, llm=llm)
    
    # generate response
    response = llm_chain.invoke({
        "column_names": df.columns,
        "target_col": target_col,  
        "model_summary": st.session_state.model_summary
    })

    response_text = textwrap.dedent(response.get("text", "")).strip()
    cleaned_text = extract_cleaned_response(response_text)
    
    return cleaned_text

# timeseries llm
def timeseries_llm(df, target_col, model):
    client = openai.OpenAI(api_key=st.session_state.openai_api_key)

    template = """
    The dataset contains time series data with the following columns:
    {column_names}

    The target column '{target_col}' has been analyzed with the following steps:

    1. **Auto ARIMA Model Summary**:
    - Below is the summary of the ARIMA model:
        ```
        {model_summary}
        ```
    - Based on this summary, provide an interpretation of the ARIMA model components and any key diagnostic statistics.

    2. **Forecasting Results**:
    - The ARIMA model predicted future values of the test set. Based on the model summary, we describe the prediction results and mention considerations.
    Answer:
    """

    # column names
    column_names = ", ".join(df.columns)
    
    # prompt template
    prompt = template.format(
        column_names=column_names,
        target_col=target_col,
        model_summary=st.session_state.model_summary
    )

    # call GPT API
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

def print_llm(df, cols):
    # show explanation
    target = st.session_state.target
    if target in cols:
        if st.session_state.llm_response:
            if st.session_state.llm_model in ["gpt-3.5-turbo", "gpt-4", "gpt-4o", "gpt-4o-mini"]:
                response = timeseries_llm(df, target, st.session_state.llm_model)
            else:
                response = timeseries_slm(df, target, st.session_state.llm_model)

            if st.session_state.language != "en":
                response = google_translate(response, "ko")

            txt = st.text_area("LLM response", response, height=200)
            st.write(f"Response: {len(txt)} characters.")
    else:
        txt = st.text_area("LLM response", "No target column", height=300)

# timeseries page
def timeseries(df, time_cols, cols):
    # title
    st.markdown("### Time Series")
    target = st.session_state.target

    # If time_cols is provided, handle time series column
    if time_cols:
        time = time_cols[0]
        if time in cols:
            df[time] = pd.to_datetime(df[time], errors='coerce')
            df.set_index(time, inplace=True)

        if isinstance(cols, list):
            if target in cols:
                cols.remove(target)
                display_timeseries(df, target)
            
            for col in cols:
                display_timeseries(df, col)
        
        print_llm(df, cols)

    else:
        # Handle case when time_cols is empty
        timeseries_index = st.session_state.get('timeseries_index', False)
        timeseries_index = st.checkbox("The index of the dataset is timeseries", value=timeseries_index)
        st.session_state.timeseries_index = timeseries_index

        if timeseries_index:
            if isinstance(cols, list):
                if target in cols:
                    cols.remove(target)
                    display_timeseries(df, target)
                
                for col in cols:
                    display_timeseries(df, col)
            
            print_llm(df, cols)
        
        else:
            st.warning(_("No Time Series Columns"))
    
###################################### String ######################################

english_stopwords = set(["the", "and", "is", "in", "to", "of", "it", "that", "for", "on", "with", "as", "this", "by", "at", "from"])

def draw_word_barchart(series):
    text = ' '.join(series.dropna().astype(str))
    english = ' '.join(re.findall(r'[a-zA-Z]+', text))
    korean = ' '.join(re.findall(r'[가-힣]+', text))

    # english word extraction without nltk
    if english:
        english = english.lower()
        tokens = re.findall(r'\b[a-zA-Z]+\b', english)
        tokens = [word for word in tokens if word not in english_stopwords]
    else:
        tokens = []

    # korean word extraction using Okt
    if korean:
        okt = Okt()
        korean_nouns = okt.nouns(korean)
    else:
        korean_nouns = []

    # get top 10 words
    total_nouns = tokens + korean_nouns
    freq_total_nouns = Counter(total_nouns)
    most_common_words = freq_total_nouns.most_common(10)
    st.session_state.most_common_words = most_common_words

    # prepare data for bar chart
    words, counts = zip(*most_common_words)

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.bar(words, counts, color='skyblue')
    ax.set_title('Top 10 Most Common Words')
    ax.set_xlabel('Words')
    ax.set_ylabel('Frequency')
    plt.xticks(rotation=45)
    plt.tight_layout()

    return fig

# string slm
def string_slm(col, model):
    # select model
    if model == "Phi3": 
        repo_id = "microsoft/Phi-3-mini-4k-instruct"

    # template 
    template = """
    The following dataset contains a column with string data. This column was analyzed to identify the most frequently used words along with their corresponding frequency counts.

    Column containing string data: {column_name}

    The top 10 frequently used words and their frequency counts are:
    {top_10_words_list}

    Please provide a detailed explanation of the frequently used words, their context in the data, and the significance of their frequency.

    Answer:
    """

    # prompt template
    prompt = PromptTemplate(
        template=template,
        input_variables=["column_name", "top_10_words_list"]
    )

    # HuggingFaceHub object
    llm = HuggingFaceHub(
        repo_id=repo_id,
        model_kwargs={"temperature": 0.2, "max_new_tokens": 1024},
    )

    # LLm Chain object
    llm_chain = LLMChain(prompt=prompt, llm=llm)
    
    # generate response
    response = llm_chain.invoke({
        "column_name": col,
        "top_10_words_list": st.session_state.most_common_words

    })

    response_text = textwrap.dedent(response.get("text", "")).strip()
    cleaned_text = extract_cleaned_response(response_text)
    
    return cleaned_text

# string llm
def string_llm(col, model):
    client = openai.OpenAI(api_key=st.session_state.openai_api_key)

    template = """
    The following dataset contains a column with string data. This column was analyzed to identify the most frequently used words along with their corresponding frequency counts.

    Column containing string data: {column_name}

    The top 10 frequently used words and their frequency counts are:
    {top_10_words_list}

    Please provide a detailed explanation of the frequently used words, their context in the data, and the significance of their frequency.

    Answer:
    """

    # get top 10 words
    top_10_words_list = st.session_state.most_common_words
    
    # Call GPT API only if string columns are found
    if col is not None:
        prompt = template.format(column_name=col, top_10_words_list=top_10_words_list)
        response = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=2024,
            temperature=0.2
        )

        answer_text = response.choices[0].message.content.strip().split("Answer:")[-1].strip()
    else:
        answer_text = "No string columns found, skipping OpenAI API call."
    
    return answer_text
 
# string page
def string(df, str_col):
    st.markdown("### String Columns")
    if len(str_col) > 0:
        for col in str_col:
            with st.expander(f"**{col}**", expanded=True):
                fig = draw_word_barchart(df[col])
                st.pyplot(fig)
    
                # show explanation
                if st.session_state.llm_response:
                    if st.session_state.llm_model in ["gpt-3.5-turbo", "gpt-4", "gpt-4o", "gpt-4o-mini"]:
                        response = string_llm(col, st.session_state.llm_model)
                    else:
                        response = string_slm(col, st.session_state.llm_model)

                    if st.session_state.language != "en":
                        response = google_translate(response, "ko")

                    txt = st.text_area("LLM response", response, height=300)
                    st.write(f"Response: {len(txt)} characters.")

    else:
        st.warning(_("No String Columns"))

