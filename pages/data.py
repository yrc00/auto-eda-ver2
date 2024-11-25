"""
This is the Data page

"""

###################################### import ######################################

# library
import streamlit as st
import pandas as pd
import gettext
import os

###################################### set ######################################

# set the current page context
st.session_state.current_page = "Data"

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

###################################### data uploader ######################################

# divide data types
def get_dtype(columns):
    categorical = ['object', 'category']
    numeric_discrete = ['int64']
    numeric_continuous = ['float64']
    datetime = ['datetime64']
    bool_ = ['bool']
    string = ['str']

    dtype_str = str(columns.dtype)

    if dtype_str in categorical:
        return 'Categorical'
    elif dtype_str in numeric_discrete:
        return 'Numeric (Discrete)'
    elif dtype_str in numeric_continuous:
        return 'Numeric (Continuous)'
    elif dtype_str in datetime:
        return 'Datetime'
    elif dtype_str in bool_:
        return 'Boolean'
    elif dtype_str in string:
        return 'String'
    else:
        return 'Other'

def file_uploader(key=None):
    with st.container(border=True):    
        # file uploader
        uploaded_file = st.file_uploader("Upload CSV file", type=["csv"], key=key)

        if uploaded_file is not None:
            try:
                # Load uploaded file
                dataset = pd.read_csv(uploaded_file)
                st.session_state.dataset = dataset

                # Get data types and create dtype_df
                dtype_dict = dataset.apply(get_dtype).to_dict()
                dtype_df = pd.DataFrame(list(dtype_dict.items()), columns=['Column', 'Data Type'])
                dtype_df.set_index('Column', inplace=True)
                st.session_state.dtype_df = dtype_df
                st.session_state.all_columns = True

                # Reset target and columns based on new file
                st.session_state.target = dataset.columns[0]
                st.session_state.columns = dataset.columns.tolist()
                st.session_state.df = dataset

                st.success("File uploaded successfully!")
                return True
                
            except pd.errors.EmptyDataError:
                st.error("The uploaded file is empty or not a valid CSV file.")
                return False
                
            except Exception as e:
                st.error(f"An error occurred: {e}")
                return False
                
        else:
            return False

###################################### target_editor ######################################

def target_editor():
    if 'df' in st.session_state:
        df = st.session_state.df
    
    # Set target column with the value from session state or default to the first column
    index = df.columns.get_loc(st.session_state.target)
    target = st.selectbox(_("Select the target column"), df.columns, index=index)
    st.session_state.target = target

###################################### columns_editor ######################################

def columns_editor():
    dataset = st.session_state.dataset
    df = st.session_state.df

    # Set "Select all columns" checkbox based on session state
    is_allcolumn = st.checkbox("Select all columns", value=st.session_state.all_columns)
    st.session_state.all_columns = is_allcolumn

    # Define columns either as all columns or selected ones
    if is_allcolumn:
        columns = df.columns.tolist()
    else:
        columns = st.multiselect(
            _("Select the columns you want to analyze"),
            df.columns.tolist(),
            st.session_state.columns
        )

    # Check if target column is selected
    if st.session_state.target not in columns:
        st.warning(_("Target column must be selected."))
    else:
        st.session_state.columns = columns

        # Update the filtered DataFrame
        st.session_state.df = dataset[st.session_state.columns]

        # Update dtype_df based on the selected columns
        dtype_dict = df.apply(get_dtype).to_dict()
        dtype_df = pd.DataFrame(list(dtype_dict.items()), columns=['Column', 'Data Type'])
        dtype_df.set_index('Column', inplace=True)
        st.session_state.dtype_df = dtype_df.loc[st.session_state.columns]

###################################### dtype_editor ######################################

# edit data type
def edit_dtype(dtype_df):
    edited_dtype_df = st.data_editor(
        dtype_df,
        column_config={
            "Data Type": st.column_config.SelectboxColumn(
                label="Data Type",
                help="Select the data type of the column",
                options=[
                    "Categorical",
                    "Numeric (Discrete)",
                    "Numeric (Continuous)",
                    "Datetime",
                    "Boolean",
                    "String",
                    "Other"
                ]
            )
        }
    )

    st.session_state.dtype_df = edited_dtype_df

# data type editor
def dtype_editor():
    if 'dataset' in st.session_state:
        dtype_df = st.session_state.dtype_df
        columns = st.session_state.columns

        st.write(_("Double click the cell to edit data type."))
        edit_dtype(dtype_df.loc[columns])
    else:
        st.warning("Please upload a CSV file to edit.")

# print preview
def data_preview():
    df = st.session_state.df

    with st.container(border=True):
        tab1, tab2 = st.tabs(['head', 'tail'])
        with tab1:
            st.dataframe(df.head(10))
        with tab2:
            st.dataframe(df.tail(10))

###################################### data_page ######################################

def data_page():
    st.title("Data")

    st.markdown("### Update Dataset")
    
    # Call file_uploader with a fixed key
    file_uploader(key="data_file_uploader")

    st.markdown("### Edit Dataset")
    if "df" in st.session_state:
        with st.container(border=True):
            col1, col2 = st.columns(2)

            with col1:
                target_editor()
                columns_editor()

            with col2:
                dtype_editor()
        
        st.markdown("### Preview")
        data_preview()
        
    else:
        st.warning(_("Please Upload a CSV File to Edit."))

###################################### main ######################################

# main page
data_page()