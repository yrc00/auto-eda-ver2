## Auto-EDA Project Version 2

This repository is dedicated to the Auto-EDA Project.

By uploading a CSV file, you can easily obtain the results of Exploratory Data Analysis (EDA).

The project includes sections for home, data, overview, visualization, correlation, modeling, report, chatbot pages. 

 - Home:description of the page
- Overview: Dataset overall analysis, missing, outliers
- Visualization: Visualization (categorical, unstructured, discrete, continuous, time series, string)
- Correlation: Visualization of correlations between features (pairplot, scatterplot graph, correlation analysis)
- Modeling: 지도학습 (RandomForest, DecisionTree, XGBoost) + Optuna, Clustering, PCA
- Report: Can print only what you want from the report
- Chatbot: Can answer questions based on uploaded csv file

This program is available in Korean and English.

## How to Use

You can view the demo on [this site](https://auto-eda-ver2.streamlit.app/)

If you want to use Auto-EDA locally, follow these steps:

**1. Clone the repository**

```
git clone https://github.com/yrc00/auto-eda.git
```

**2. Create a virtual environment using Anaconda**

```
conda create -n [environment_name] python=3.9
conda activate [environment_name]
```

**3. Install the required libraries from requirements.txt**

```
pip install -r requirements.txt
```

**4.Run the Streamlit app**

```
streamlit run streamlit_app.py
```


**5. Edit Code**
If you want to type the api key in the sidebar, you don't need to modify the source code.

If you want to specify an api key and use it, follow the steps below to modify the code.

```
# .streamlit/secrets.toml
HUGGINGFACEHUB_API_TOKEN="yur_huggingface_api_key_here"
OPENAI_API_KEY="your_openai_api_key_here"
```
Enter your api key on the secrets.toml

```
# sidebar
def llm_model_selector():
    ...
    # Show API key input fields based on the selected model
    if selected_model in ["gpt-3.5-turbo", "gpt-4", "gpt-4o", "gpt-4o-mini"]:
        # OpenAI API key input
        openai_api_key = st.sidebar.text_input(
            "Enter OpenAI API Key:",
            type="password",
            value=st.session_state.get("openai_api_key", ""),
        )

        # Update OpenAI API key
        if openai_api_key != st.session_state.get("openai_api_key", ""):
            st.session_state.openai_api_key = openai_api_key
            os.environ['OPENAI_API_KEY'] = openai_api_key
            st.sidebar.success("OpenAI API Key updated successfully!")
    else:
        # HuggingFace API key input
        huggingface_api_key = st.sidebar.text_input(
            "Enter HuggingFace API Key:",
            type="password",
            value=st.session_state.get("huggingface_api_key", ""),
        )

        # Update HuggingFace API key
        if huggingface_api_key != st.session_state.get("huggingface_api_key", ""):
            st.session_state.huggingface_api_key = huggingface_api_key
            os.environ['HUGGINGFACEHUB_API_TOKEN'] = huggingface_api_key
            st.sidebar.success("HuggingFace API Key updated successfully!")

```
Annotate the code part above

```
#################### get API key from secrets.toml ####################

# # HuggingFace API token
# huggingface_api_token = st.secrets.get("HUGGINGFACEHUB_API_TOKEN")

# if huggingface_api_token:
#     os.environ['HUGGINGFACEHUB_API_TOKEN'] = huggingface_api_token
# else:
#     st.error(_("HUGGINGFACEHUB_API_TOKEN is missing. Please check your secrets.toml file."))

# # OpenAI API key
# openai_api_key = st.secrets.get("OPENAI_API_KEY")

# if openai_api_key:
#     os.environ['OPENAI_API_KEY'] = openai_api_key
# else:
#     st.error(_("OPENAI_API_KEY is missing. Please check your secrets.toml file."))

#################### get API key from sidebar ####################

# huggingface API token
if "llm_model" in st.session_state:
    if st.session_state.llm_model in ["Phi3"]:
        api_key = st.session_state.get("huggingface_api_key", "")
    else:
        api_key = st.session_state.get("openai_api_key", "") 
```
On the overview, visualization, correlation, and chatbot pages, turn off the annotation of the get API key from secrets.toml portion, and annotate the get API key from sidebar portion.

---
## Auto-EDA Project Version 2

Auto EDA Version 2 프로젝트를 위한 레포지토리입니다. 

CSV 파일을 업로드해서 탐색적 데이터 분석(EDA)의 결과를 쉽게 얻을 수 있습니다. 

해당 프로젝트는 home, data, overview, visualization, correlation, modeling, report, chatbot 페이지로 구성됩니다. 

- Home: 페이지에 대한 설명
- Overview: 데이터셋의 전반적인 분석, 결측치, 이상치
- Visualization: 시각화 (범주형, 불형, 이산형, 연속형, 시계열, 문자열)
- Correlation: 피처 간 상관관계 시각화 (pairplot, 산점도 그래프, 상관관계 분석)
- Modeling: 지도학습 (RandomForest, DecisionTree, XGBoost) + Optuna, Clustering, PCA
- Report: 보고서 내용 중 원하는 내용만 출력 가능
- Chatbot: 업로드된 csv 파일을 기반으로 질의응답 가능

이 프로그램은 한국어와 영어로 제공됩니다.

## How to Use

[이 페이지](https://auto-eda-ver2.streamlit.app/)에서 데모 페이지를 확인할 수 있습니다.

Auto EDA를 로컬 환경에서 사용하고 싶다면 아래 과정을 따라주세요:

**1. 레포지토리 클론**

```
git clone https://github.com/yrc00/auto-eda.git
```

**2. 아나콘다에서 가상환경 생성**

```
conda create -n [environment_name] python=3.9
conda activate [environment_name]
```

**3. requirements.txt에 있는 라이브러리 설치**

```
pip install -r requirements.txt
```

**4.Streamlit app 실행**

```
streamlit run streamlit_app.py
```


**5. 코드 수정**
사이드바에서 api key를 입력하여 사용하고 싶다면 코드를 수정할 필요가 없음

만약 특정 api key를 고정하여 사용하고 싶다면 아래 과정을 따라 코드를 수정하세요

```
# .streamlit/secrets.toml
HUGGINGFACEHUB_API_TOKEN="yur_huggingface_api_key_here"
OPENAI_API_KEY="your_openai_api_key_here"
```
secrets.toml 페이지에 api 키를 입력

```
# sidebar
def llm_model_selector():
    ...
    # Show API key input fields based on the selected model
    if selected_model in ["gpt-3.5-turbo", "gpt-4", "gpt-4o", "gpt-4o-mini"]:
        # OpenAI API key input
        openai_api_key = st.sidebar.text_input(
            "Enter OpenAI API Key:",
            type="password",
            value=st.session_state.get("openai_api_key", ""),
        )

        # Update OpenAI API key
        if openai_api_key != st.session_state.get("openai_api_key", ""):
            st.session_state.openai_api_key = openai_api_key
            os.environ['OPENAI_API_KEY'] = openai_api_key
            st.sidebar.success("OpenAI API Key updated successfully!")
    else:
        # HuggingFace API key input
        huggingface_api_key = st.sidebar.text_input(
            "Enter HuggingFace API Key:",
            type="password",
            value=st.session_state.get("huggingface_api_key", ""),
        )

        # Update HuggingFace API key
        if huggingface_api_key != st.session_state.get("huggingface_api_key", ""):
            st.session_state.huggingface_api_key = huggingface_api_key
            os.environ['HUGGINGFACEHUB_API_TOKEN'] = huggingface_api_key
            st.sidebar.success("HuggingFace API Key updated successfully!")

```
위의 코드를 주석처리

```
#################### get API key from secrets.toml ####################

# # HuggingFace API token
# huggingface_api_token = st.secrets.get("HUGGINGFACEHUB_API_TOKEN")

# if huggingface_api_token:
#     os.environ['HUGGINGFACEHUB_API_TOKEN'] = huggingface_api_token
# else:
#     st.error(_("HUGGINGFACEHUB_API_TOKEN is missing. Please check your secrets.toml file."))

# # OpenAI API key
# openai_api_key = st.secrets.get("OPENAI_API_KEY")

# if openai_api_key:
#     os.environ['OPENAI_API_KEY'] = openai_api_key
# else:
#     st.error(_("OPENAI_API_KEY is missing. Please check your secrets.toml file."))

#################### get API key from sidebar ####################

# huggingface API token
if "llm_model" in st.session_state:
    if st.session_state.llm_model in ["Phi3"]:
        api_key = st.session_state.get("huggingface_api_key", "")
    else:
        api_key = st.session_state.get("openai_api_key", "") 
```
overview, visualization, correlation, chatbot page에서 get API key from secrets.toml 아래의 주석처리를 제거하고, get API key from sidebar 아래를 주석처리
