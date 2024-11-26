"""
This is the Chatbot page

"""

###################################### import ######################################

# library
from typing import List, Union
from langchain_experimental.agents.agent_toolkits import create_pandas_dataframe_agent
from langchain_experimental.tools import PythonAstREPLTool
from langchain_openai import ChatOpenAI
from messages import AgentStreamParser, AgentCallbacks
from dotenv import load_dotenv
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import os

###################################### set  ######################################

# set the current page context
st.session_state.current_page = "Chatbot"

#################### get API key from secrets.toml ####################

# # openai API key
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

############################ Streamlit session state initialization ############################

# Initialize session state - messages
if "messages" not in st.session_state:
    st.session_state["messages"] = []

# Constants
class MessageRole:
    """
    Defines the message roles.
    """
    USER = "user"           # User message role
    ASSISTANT = "assistant" # Assistant message role


class MessageType:
    """
    Defines the message types.
    """
    TEXT = "text"           # Text message
    FIGURE = "figure"       # Figure message
    CODE = "code"           # Code message
    DATAFRAME = "dataframe" # DataFrame message

############################ Message related functions ############################

def print_messages():
    """
    Prints the saved messages to the screen.
    """
    for role, content_list in st.session_state["messages"]:
        with st.chat_message(role):
            for content in content_list:
                if isinstance(content, list):
                    message_type, message_content = content
                    if message_type == MessageType.TEXT:        # Text message
                        st.markdown(message_content)
                    elif message_type == MessageType.FIGURE:    # Figure message
                        st.pyplot(message_content)
                    elif message_type == MessageType.CODE:      # Code message
                        with st.status("Code Output", expanded=False):
                            st.code(message_content, language="python")
                    elif message_type == MessageType.DATAFRAME: # DataFrame message
                        st.dataframe(message_content)
                else:
                    raise ValueError(f"Unknown content type: {content}") # Unknown content type

def add_message(role: MessageRole, content: List[Union[MessageType, str]]):
    """
    Adds a new message to the saved messages.

    ARgs:
        role (MessageRole): Message role (user or assistant)
        content (List[Union[MessageType, str]]): Message content
    """
    messages = st.session_state["messages"]
    if messages and messages[-1][0] == role: # Combine consecutive messages of the same role
        messages[-1][1].extend([content])
    else:
        messages.append([role, [content]]) # Add new message

############################ Callback functions ############################

def tool_callback(tool) -> None:
    """
    Callback function to process the tool execution results.

    Args:
        tool (dict): Tool information
    """
    if tool_name := tool.get("tool"):
        if tool_name == "python_repl_ast":
            tool_input = tool.get("tool_input", {})
            query = tool_input.get("query")
            if query: 
                df_in_result = None
                with st.status("Data analysis in progress...", expanded=True) as status:
                    st.markdown(f"```python\n{query}\n```")
                    add_message(MessageRole.ASSISTANT, [MessageType.CODE, query])
                    if "df" in st.session_state:
                        result = st.session_state["python_tool"].invoke({"query": query})
                        if isinstance(result, pd.DataFrame):
                            df_in_result = result
                    status.update(label = "Print Code", state="complete", expanded=False)
                
                if df_in_result is not None: 
                    st.dataframe(df_in_result)
                    add_message(MessageRole.ASSISTANT, [MessageType.DATAFRAME, df_in_result])

                if "plt.show" in query:
                    fig = plt.gcf()
                    st.pyplot(fig)
                    add_message(MessageRole.ASSISTANT, [MessageType.FIGURE, fig])
                
                return result
            else:
                st.error("DataFrame is not defined. Please upload a CSV file first.")
                return
        else:
            st.error(f"Unknown tool: {tool_name}")
            return
    else:
        st.error("Tool name is not defined.")
        return

def observation_callback(observation) -> None:
    """
    Callback function to process the observation results.

    Args:
        observation (dict): Observation results
    """
    if "observation" in observation:
        obs = observation["observation"]
        if isinstance(obs, str) and "Error" in obs:
            st.error(obs)
            st.session_state["messages"][-1][1].clear() # Delete the last message if an error occurs

def result_callback(result: str) -> None:
    """
    Callback function to process the final result.

    Args:
        result (str): Final result
    """
    pass # Currently does nothing

############################ Agent creation function ############################

def create_agent(dataframe, selected_model="gpt-4o", language="en"):
    """
    Creates a DataFrame agent for the given DataFrame.

    Args:
        dataframe (pd.DataFrame): DataFrame to analyze
        selected_model (str, optional): Selected OpenAI model. Defaults to "gpt-4o".
    
    Returns:
        Agent: Created DataFrame agent
    """

    lan_map = {"en": "english", "ko": "korean"}
    if language in lan_map:
        language = lan_map[language]

    return create_pandas_dataframe_agent(
        ChatOpenAI(model=selected_model, temperature=0, openai_api_key=api_key),
        dataframe,
        verbose=False,
        agent_type="tool-calling",
        allow_dangerous_code=True,
        prefix="You are a professional data analyst and expert in Pandas. "
        "You must use Pandas DataFrame(`df`) to answer user's request. "
        "\n\n[IMPORTANT] DO NOT create or overwrite the `df` variable in your code. \n\n"
        "If you are willing to generate visualization code, please use `plt.show()` at the end of your code. "
        "I prefer seaborn code for visualization, but you can use matplotlib as well."
        "\n\n<Visualization Preference>\n"
        "- [IMPORTANT] Use `English` for your visualization title and labels."
        "- `muted` cmap, white background, and no grid for your visualization."
        "\nRecommend to set cmap, palette parameter for seaborn plot if it is applicable. "
        f"The language of final answer should be written in {language}. "
        "\n\n###\n\n<Column Guidelines>\n"
        "If user asks with columns that are not listed in `df.columns`, you may refer to the most similar columns listed below.\n",
    )

############################ Question processing function ############################

def ask(query):
    """
    Ask Function for processing user questions and generating responses.

    Args:
        query (str): User question
    """
    if "agent" in st.session_state:
        st.chat_message("user").write(query)
        add_message(MessageRole.USER, [MessageType.TEXT, query])

        agent = st.session_state["agent"]
        response = agent.stream({"input": query})

        ai_answer = ""
        parser_callback = AgentCallbacks(
            tool_callback, observation_callback, result_callback
        )
        stream_parser = AgentStreamParser(parser_callback)

        with st.chat_message("assistant"):
            for step in response:
                stream_parser.process_agent_steps(step)
                if "output" in step:
                    ai_answer += step["output"]
            st.write(ai_answer)

        add_message(MessageRole.ASSISTANT, [MessageType.TEXT, ai_answer])

############################ Streamlit code for page ############################


def chatbot_logic():
    """
    Chatbot logic for the Streamlit page.
    """
    # 사이드바에 Start Chat 및 Clear Chat 버튼
    with st.sidebar:
        col1, col2 = st.columns(2)
        with col1:
            apply_btn = st.button("Start Chat")
        with col2:
            clear_btn = st.button("Clear Chat")
        
    if clear_btn:
        st.session_state["messages"] = []

    if apply_btn:
        try:
            loaded_data = st.session_state.df
            st.session_state["python_tool"] = PythonAstREPLTool()
            st.session_state["python_tool"].locals["df"] = loaded_data
            st.session_state["agent"] = create_agent(
                loaded_data, 
                st.session_state.llm_model, 
                st.session_state.language
            )
            st.session_state.apply_btn = True
            st.success("Settings are complete. Please start the conversation!")
        except ValueError as e:
            st.error(f"Failed to load the dataset: {e}")

    if not st.session_state.apply_btn:
        st.warning("Please click the 'Start Chat' to start the chatbot.")

    print_messages()

    user_input = st.chat_input("Ask me anything!")
    if user_input:
        ask(user_input)

def chatbot_page():
    """
    Main page of the chatbot.
    """
    st.title("Chatbot")

    if "messages" not in st.session_state:
        st.session_state["messages"] = []  # Initialize messages
    if not st.session_state.get("openai_api_key") and not st.secrets.get("OPENAI_API_KEY"):
        st.warning("Please enter your OpenAI API Key in the sidebar to use the chatbot.")
    elif "df" not in st.session_state:
        st.warning("Please upload a CSV file to start the chatbot.")
    else:
        chatbot_logic()

###################################### main  ######################################
# main page
chatbot_page()