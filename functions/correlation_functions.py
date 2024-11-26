"""
This is the Correlation page

"""

###################################### import ######################################

# library
from dotenv import load_dotenv
import gettext
import os

import streamlit as st
import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from scipy.stats import shapiro, skew, kurtosis

import re
from langchain import LLMChain
from langchain.prompts import PromptTemplate
from langchain.llms import HuggingFaceHub
from deep_translator import GoogleTranslator
import textwrap
import openai

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

##################################### Pairplot ######################################

# draw pairplot
@st.cache_data
def get_pairplot(df, target):
    if target is None:
        return sns.pairplot(df, height=2)
    else:
        return sns.pairplot(df, hue=target, height=2)

# pairplot preprocessing
def normality_test(df):
    """
    Perform Shapiro-Wilk normality test on each column of the DataFrame.
    """
    alpha = 0.05
    results = []

    # Shapiro-Wilk test
    for column in df.columns:
        data = df[column].dropna()

        if len(data) < 3:
            results.append([column, None, None, 'Not Enough Data'])
            continue

        try:
            stat, p = stats.shapiro(data)
            normal = 'Normal' if p > alpha else 'Not Normal'
            results.append([column, stat, p, normal])
        except Exception as e:
            results.append([column, None, None, f'Error: {str(e)}'])

    result_df = pd.DataFrame(results, columns=['Column', 'Shapiro-Wilk Statistics', 'p-value', 'Normality'])
    return result_df

def correlation(df):
    """
    Calculate correlation coefficients using Pearson, Spearman, and Kendall methods.
    """
    pearson = df.corr(method='pearson')
    spearman = df.corr(method='spearman')
    kendall = df.corr(method='kendall')
    return pearson, spearman, kendall

def find_top_linear_column_pairs(df, top_n=5):
    """
    Based on normality test results, choose the appropriate correlation method 
    and find the top n linear column pairs. Adds covariance and R-squared.
    """
    # Perform normality test
    normality_results = normality_test(df)

    # Identify columns that follow a normal distribution
    normal_columns = normality_results[normality_results['Normality'] == 'Normal']['Column'].tolist()

    # Choose the appropriate correlation method based on normality of columns
    if len(normal_columns) == len(df.columns):
        method = 'pearson'
    else:
        method = 'spearman'

    # Calculate correlation matrix using the selected method
    corr_matrix = df.corr(method=method)

    # Exclude self-correlation and sort by absolute value
    mask = np.triu(np.ones_like(corr_matrix, dtype=bool), k=1)
    corr_matrix = corr_matrix.where(mask)
    corr_pairs = corr_matrix.unstack().dropna().abs().sort_values(ascending=False)

    # Exclude identical column pairs and extract top n pairs
    top_pairs = corr_pairs.head(top_n).reset_index()
    top_pairs.columns = ['Column 1', 'Column 2', 'Correlation']

    # Add covariance and R-squared values
    covariances = []
    r_squared_values = []
    for _, row in top_pairs.iterrows():
        col1 = df[row['Column 1']]
        col2 = df[row['Column 2']]
        cov = np.cov(col1, col2)[0, 1]
        r_squared = row['Correlation'] ** 2  # Calculate R^2
        covariances.append(cov)
        r_squared_values.append(r_squared)

    top_pairs['Covariance'] = covariances
    top_pairs['R^2'] = r_squared_values

    return top_pairs, method

def calculate_skew_kurtosis(df):
    """
    Calculate skewness and kurtosis for each column in the DataFrame and return the results.
    """
    results = []
    for column in df.columns:
        col_skew = skew(df[column].dropna())
        col_kurt = kurtosis(df[column].dropna())
        results.append({
            'Column': column,
            'Skewness': col_skew,
            'Kurtosis': col_kurt
        })
    return pd.DataFrame(results)

# pairplot explain
def pairplot_slm(model, df):
    # model
    if model == "Phi3": 
        repo_id = "microsoft/Phi-3-mini-4k-instruct"

    # template 
    template = """
    The dataset analyzed contains the following columns: {column_names}.
    A pairplot was generated to provide insights into the relationships between these columns. Please give a general interpretation of the pairplot, focusing on the overall trends, notable relationships, and any patterns between variables.

    The top 5 column pairs with the strongest linear relationships are: {linear_columns}.
    The correlation was calculated using the {method_used} method, with additional analysis including covariance and \( R^2 \) values to describe the linear strength between pairs.
    Each column's skewness and kurtosis values are also provided here: {histogram}, describing the shape and symmetry of each variable's distribution.

    Summarize the pairplot based on the linear relationships, distribution shapes, and any other key observations about how the variables relate to one another. Avoid giving specific correlation or covariance values, but focus on trends, relationships, and general distribution characteristics.

    Please provide a thorough summary of the pairplot and variable relationships.

    Answer:
    """

    # prompt template
    prompt = PromptTemplate(
        template=template,
        input_variables=["column_names", "linear_columns", "method_used", "histogram"]
    )

    # HuggingFaceHub object
    llm = HuggingFaceHub(
        repo_id=repo_id,
        model_kwargs={"temperature": 0.2, 
                    "max_new_tokens" : 2048}
    )

    # LLm Chain object
    llm_chain = LLMChain(prompt=prompt, llm=llm)

    column_names = ', '.join(df.columns)
    linear_columns, method_used = find_top_linear_column_pairs(df)
    histogram = calculate_skew_kurtosis(df)

   # generate response
    response = llm_chain.invoke({
        "column_names": column_names,
        "linear_columns": linear_columns.to_string(index=False),
        "method_used": method_used,
        "histogram": histogram
    })

    response_text = textwrap.dedent(response.get("text", "")).strip()
    cleaned_text = extract_cleaned_response(response_text)

    # "Answer:"가 없는 경우 처리
    description_text = "Pairplot Summary:\n\n" + cleaned_text
    return description_text

def pairplot_llm(model, df):
    client = openai.OpenAI(api_key=st.session_state.openai_api_key)

    template = """
    The dataset analyzed contains the following columns: {column_names}.
    A pairplot was generated to provide insights into the relationships between these columns. Please give a general interpretation of the pairplot, focusing on the overall trends, notable relationships, and any patterns between variables.

    The top 5 column pairs with the strongest linear relationships are: {linear_columns}.
    The correlation was calculated using the {method_used} method, with additional analysis including covariance and \( R^2 \) values to describe the linear strength between pairs.
    Each column's skewness and kurtosis values are also provided here: {histogram}, describing the shape and symmetry of each variable's distribution.

    Summarize the pairplot based on the linear relationships, distribution shapes, and any other key observations about how the variables relate to one another. Avoid giving specific correlation or covariance values, but focus on trends, relationships, and general distribution characteristics.

    Please provide a thorough summary of the pairplot and variable relationships.

    Answer:
    """

    column_names = ', '.join(df.columns)
    linear_columns, method_used = find_top_linear_column_pairs(df)
    histogram = calculate_skew_kurtosis(df)

    prompt = template.format(column_names=column_names, linear_columns=linear_columns, method_used=method_used, histogram=histogram) 
    response = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        max_tokens=2024,
        temperature=0.2
    )

    answer_text = response.choices[0].message.content.strip().split("Answer:")[-1].strip()
    return answer_text

# pairplot page
def pairplot(df):
    # title
    st.markdown("### Pairplot")
    target = st.session_state.target

    with st.container(border=True):
        if target not in df.columns:
            pairplot_fig = get_pairplot(df, None)
        else:
            pairplot_fig = get_pairplot(df, target)
        st.pyplot(pairplot_fig)

    # show explanation
    if st.session_state.llm_response:
        if st.session_state.llm_model in ["gpt-3.5-turbo", "gpt-4", "gpt-4o", "gpt-4o-mini"]:
            response = pairplot_llm(st.session_state.llm_model, df)
        else:
            response = pairplot_slm(st.session_state.llm_model, df)

        if st.session_state.language != "en":
            response = google_translate(response, "ko")

        txt = st.text_area("LLM response", response, height=500)
        st.write(f"Response: {len(txt)} characters.")

##################################### Scatter Plot ######################################

# draw scatter plot
@st.cache_data
def get_scatter(df, x, y):
    fig, ax = plt.subplots(figsize=(10, 10))
    sns.scatterplot(data=df, x=x, y=y, ax=ax)
    return fig

# summary dataframe
def generate_summary_dataframe(df, x_column, y_column):
    # select x, y columns
    x_data = df[x_column]
    y_data = df[y_column]

    # summary statistics
    summary_data = {
        "Metric": ["Mean", "Standard Deviation", "Min", "Max", "Correlation"],
        "X": [
            x_data.mean(),
            x_data.std(),
            x_data.min(),
            x_data.max(),
            x_data.corr(y_data)
        ],
        "Y": [
            y_data.mean(),
            y_data.std(),
            y_data.min(),
            y_data.max(),
            x_data.corr(y_data)
        ]
    }

    # dataframe
    summary_df = pd.DataFrame(summary_data)
    return summary_df

# scatter plot explanation
def scatter_slm(df, x, y, model):
    # model
    if model == "Phi3": 
        repo_id = "microsoft/Phi-3-mini-4k-instruct"

    # template
    template = """
    The dataset used contains the following columns:
    {column_names}

    A scatter plot was generated using the X-axis: {x_column} and Y-axis: {y_column}. Please provide a detailed explanation of this scatter plot. 

    Describe the general appearance of the plot, including any noticeable patterns, clusters, or outliers. 

    Additionally, explain the potential relationships between the variables on the X and Y axes and any insights that can be inferred from the scatter plot.

    Answer:
    """

    # prompt template
    prompt = PromptTemplate(
        template=template,
        input_variables=["column_names", "x_column", "y_column"]
    )

    # HuggingFaceHub object
    llm = HuggingFaceHub(
        repo_id=repo_id,
        model_kwargs={"temperature": 0.2, 
                    "max_new_tokens" : 2048}
    )

    # generate prompt template
    prompt = PromptTemplate(
        template=template,
        input_variables=["column_names", "x_column", "y_column"]
    )

    # LLM Chain object
    llm_chain = LLMChain(prompt=prompt, llm=llm)

    x_column = x
    y_column = y

    if x_column and y_column:
        # 열 이름 목록
        column_names = ', '.join(df.columns)

        # generate response
        response = llm_chain.invoke({
            "column_names": column_names,
            "x_column": x_column,
            "y_column": y_column
        })

        response_text = textwrap.dedent(response.get("text", "")).strip()
        cleaned_text = extract_cleaned_response(response_text)

        description_text = "Summarize Scatterplot:\n\n" + cleaned_text
        return description_text
    
    else:
        return "Please select X and Y columns to generate a scatter plot."

def scatter_llm(df, x, y, model):
    client = openai.OpenAI(api_key = st.session_state.openai_api_key)

    template = """
    The dataset used contains the following columns:
    {column_names}

    A scatter plot was generated using the X-axis: {x_column} and Y-axis: {y_column}. Please provide a detailed explanation of this scatter plot.

    Describe the general appearance of the plot, including any noticeable patterns, clusters, or outliers.

    Additionally, explain the potential relationships between the variables on the X and Y axes and any insights that can be inferred from the scatter plot.

    Answer:
    """

    x_column = x
    y_column = y

    if x_column and y_column:
        # 열 이름 목록
        column_names = ', '.join(df.columns)

        prompt = template.format(column_names=column_names, x_column=x_column, y_column=y_column)
        response = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=2024,
            temperature=0.2
        )

        answer_text = response.choices[0].message.content.strip().split("Answer:")[-1].strip()
        return answer_text
    else:
        return "Please select X and Y columns to generate a scatter plot."


# scatter plot page
def scatter(df):
    # title
    st.markdown("### Scatter Plot")

    # input x, y
    col1, col2 = st.columns(2)
    with col1:
        x = st.selectbox("X", df.columns)
    with col2:
        y = st.selectbox("Y", df.columns)
    
    # plot
    with st.container(border=True):
        scatter_fig = get_scatter(df, x, y)
        st.pyplot(scatter_fig)
    
    # show explanation
    if st.session_state.llm_response:
        if st.session_state.llm_model in ["gpt-3.5-turbo", "gpt-4", "gpt-4o", "gpt-4o-mini"]:
            response = scatter_llm(df, x, y, st.session_state.llm_model)
        else:
            response = scatter_slm(df, x, y, st.session_state.llm_model)

        if st.session_state.language != "en":
            response = google_translate(response, "ko")

        txt = st.text_area("LLM response", response, height=500)
        st.write(f"Response: {len(txt)} characters.")

##################################### Correlation Heatmap ######################################

# normality test
def correlation_normality(df):
    if "normaliy_df" in st.session_state:
        result_df = st.session_state.normality_df
    else:
        result_df = normality_test(df)
    
    # highlight p-value
    def highlight_p_value(val):
        if val is None:  # None 값 처리 추가
            return ''
        color = 'lightyellow' if val < 0.05 else ''
        return f'background-color: {color}'

    # show result
    st.dataframe(result_df.style.applymap(highlight_p_value, subset=['p-value']))

# heatmap description
def heatmap_description(model):
    # model
    if model == "Phi3":
        repo_id = "microsoft/Phi-3-mini-4k-instruct"
    
    # template 
    template = """
    Define the following data analysis terms in short sentences:

    - Correlation:

    - Pearson Correlation:

    - Spearman Correlation:

    - Kendall Correlation:

    - Heatmap:

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
                    "max_new_tokens" : 200}
    )

    # LLM Chain object
    llm_chain = LLMChain(prompt=prompt, llm=llm)

    # generate response
    response = llm_chain.invoke({})

    response_text = textwrap.dedent(response.get("text", "")).strip()
    cleaned_text = extract_cleaned_response(response_text)
    
    description_text = "Definition of Correlation Heatmap:\n\n" + cleaned_text

    return description_text

# heatmap explanation
def select_correlation_method(normal_ratio, rows, df_numeric):
    # Assuming correlation is a previously defined function that returns three correlation matrices
    pearson, spearman, kendall = correlation(df_numeric)
    if normal_ratio >= 1.0:
        method = pearson
        method_name = "Pearson"
    else:
        if rows < 30:
            method = kendall
            method_name = "Kendall"
        else:
            method = spearman
            method_name = "Spearman"
    return method, method_name

def heatmap_slm(df, model):
    # model
    if model == "Phi3": 
        repo_id = "microsoft/Phi-3-mini-4k-instruct"

    # template
    template = """
    {normal_ratio} of the columns in the dataset passed the normality test.
    Here is the result of the {method} correlation matrix:
    {correlation_table}

    Please analyze and explain the insights based on the correlation matrix:
    
    Target variable:
    - Focus specifically on the correlation coefficients between the target varaible ({target}) and all other variables.
    - Provide the list of columns with meaningful correlation(having correlation coefficient greater than 0.4 or less than -0.4) with the target variable.

    Strong correlations:
    - Provide a detailed interpretation of the relationships between the columns based on the column name
    - Strong positive correlations: coefficients greater than 0.8.
    - Strong negative correlations: coefficients less than -0.8.
    - Describe how the relationships between the target and the other variables may indicate trends, patterns, or insights from the data.
    - Avoid discussing the entire correlation matrix; focus only on the most relevant correlations based on the criteria above.


    Medium correlations:
    - Provide a detailed interpretation of the relationships between the columns based on the column name
    - Medium positive correlations: coefficients between 0.4 and 0.8.
    - Medium negative correlations: coefficients between -0.4 and -0.8.
    - Describe how the relationships between the target and the other variables may indicate trends, patterns, or insights from the data.
    - Avoid discussing the entire correlation matrix; focus only on the most relevant correlations based on the criteria above.

    Answer:
    """

    # prompt template
    prompt = PromptTemplate(
        template=template,
        input_variables=["normal_ratio", "method", "correlation_table", "target"]
    )

    # HuggingFaceHub object
    llm = HuggingFaceHub(
        repo_id=repo_id,
        model_kwargs={"temperature": 0.2, 
                    "max_new_tokens" : 2048}
    )

    # LLM Chain object
    llm_chain = LLMChain(prompt=prompt, llm=llm)

    # generate response
    normality = normality_test(df)
    normal_ratio = (normality['Normality'] == 'Normal').mean()
    method, method_name = select_correlation_method(normal_ratio, len(df), df)
    response = llm_chain.invoke({
        "normal_ratio": normal_ratio,
        "method": method_name,
        "correlation_table": method,
        "target": "Survived"
    })

    response_text = textwrap.dedent(response.get("text", "")).strip()
    cleaned_text = extract_cleaned_response(response_text)

    description_text = "\n\nAbout Correlation Heatmap:\n\n" + cleaned_text
    return description_text

def heatmap_llm(df, model):
    client = openai.OpenAI(api_key=st.session_state.openai_api_key)

    template = """
    {normal_ratio} of the columns in the dataset passed the normality test.
    Here is the result of the {method} correlation matrix:
    {correlation_table}

    Please analyze and explain the insights based on the correlation matrix:
    1. Focus specifically on the correlation coefficients between the target variable (alcohol content) and all other variables.
    2. Highlight and explain any strong correlations:
    - Strong positive correlations: coefficients greater than 0.5.
    - Strong negative correlations: coefficients less than -0.5.
    3. Provide a detailed interpretation of these strong correlations:
    - Describe how the relationships between the target and the other variables may indicate trends, patterns, or insights from the data.
    - Avoid discussing the entire correlation matrix; focus only on the most relevant correlations based on the criteria above.
    
    Answer:
    """

    normality = normality_test(df)
    normal_ratio = (normality['Normality'] == 'Normal').mean()
    method, method_name = select_correlation_method(normal_ratio, len(df), df)
    prompt = template.format(normal_ratio=normal_ratio, method=method_name, correlation_table=method)

    response = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        max_tokens=2024,
        temperature=0.2
    )

    answer_text = response.choices[0].message.content.strip().split("Answer:")[-1].strip()
    return answer_text

# heatmap
@st.cache_data
def plot_heatmap(df, correlation_method):
    cor_matrix = df.corr(method=correlation_method)
    mask = np.triu(np.ones_like(cor_matrix, dtype=bool), k=1)
    cor_figure = plt.figure(figsize=(10, 10))
    sns.heatmap(cor_matrix, annot=True, mask=mask, fmt=".2f", cmap='coolwarm', square=True)
    st.pyplot(cor_figure)

# correlation heatmap page
def heatmap(df):
    # title
    st.markdown("### Correlation Heatmap")
    with st.container(border=True):
        st.markdown("**Normality Test**")
        correlation_normality(df)


    with st.container(border=True):
        st.markdown("**Correlation Heatmap**")
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
    
    # show explanation
    if st.session_state.llm_response:
        if st.session_state.llm_model in ["gpt-3.5-turbo", "gpt-4", "gpt-4o", "gpt-4o-mini"]:
            response = heatmap_llm(df, st.session_state.llm_model)
        else:
            description_txt = heatmap_description(st.session_state.llm_model)
            analyze_txt = heatmap_slm(df, st.session_state.llm_model)
            response = description_txt + analyze_txt

        if st.session_state.language != "en":
            response = google_translate(response, "ko")
        
        txt = st.text_area("LLM response", response, height=500)
        st.write(f"Response: {len(txt)} characters.")


