import streamlit as st
import os
import pandas as pd
import sqlite3
from dotenv import load_dotenv
from sqlalchemy import create_engine

# --- LangChain / LangGraph imports ---
from langchain_community.utilities import SQLDatabase
from langchain_community.agent_toolkits import SQLDatabaseToolkit
from langchain_openai import AzureChatOpenAI

from langchain_core.messages import AIMessage, ToolMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableLambda, RunnableWithFallbacks
from langchain_core.tools import tool

from langgraph.graph import END, StateGraph, START
from langgraph.graph.message import AnyMessage, add_messages
from langgraph.prebuilt import ToolNode
from typing_extensions import TypedDict
from pydantic import BaseModel, Field
from typing import Annotated, Literal, Any

# ----------------------------------------------------------------
# 1) Load environment variables
# ----------------------------------------------------------------
load_dotenv()

AZURE_OPENAI_API_KEY = os.getenv("AZURE_OPENAI_API_KEY")
AZURE_OPENAI_ENDPOINT = os.getenv("AZURE_OPENAI_ENDPOINT")
AZURE_OPENAI_DEPLOYMENT = os.getenv("AZURE_OPENAI_DEPLOYMENT")
AZURE_OPENAI_API_VERSION = os.getenv("AZURE_OPENAI_API_VERSION")

# Check if credentials are present
if not all([AZURE_OPENAI_API_KEY, AZURE_OPENAI_ENDPOINT, AZURE_OPENAI_DEPLOYMENT]):
    st.error("Azure OpenAI API credentials are missing. Please check your .env file.")


# ----------------------------------------------------------------
# 2) CSV/DB loading helpers - Configure the database
# ----------------------------------------------------------------
def load_csv_to_file_db(uploaded_csv) -> str:
    """
    Save the CSV to a file-based SQLite database `temp_uploaded_db.sqlite`.
    Return the absolute path to that DB.
    """
    try:
        # Save uploaded CSV to a local file (to read it with pandas)
        temp_csv_path = "temp_uploaded_csv.csv"
        with open(temp_csv_path, "wb") as f:
            f.write(uploaded_csv.read())

        # Now read with pandas
        df = pd.read_csv(temp_csv_path)
        # Overwrite our file-based DB (so it’s consistent each time)
        db_path = "temp_uploaded_db.sqlite"
        engine = create_engine(f"sqlite:///{db_path}")
        df.to_sql("UploadedTable", engine, if_exists="replace", index=False)
        return db_path
    except Exception as e:
        st.error(f"Failed to load CSV: {e}")
        return ""

def load_db_file(uploaded_db) -> str:
    """
    Save user-uploaded .db or .sqlite file to local disk at `temp_uploaded_db.sqlite`,
    and return its path.
    """
    try:
        db_path = "temp_uploaded_db.sqlite"
        with open(db_path, "wb") as f:
            f.write(uploaded_db.read())
        return db_path
    except Exception as e:
        st.error(f"Failed to load SQLite database: {e}")
        return ""

def build_sql_database_from_path(db_path: str) -> SQLDatabase:
    """Construct a SQLDatabase from a file-based path."""
    engine = create_engine(f"sqlite:///{db_path}")
    return SQLDatabase(engine=engine)



# ----------------------------------------------------------------
# 3) Utility: fallback handler for tool errors
# ----------------------------------------------------------------
def create_tool_node_with_fallback(tools: list) -> RunnableWithFallbacks[Any, dict]:
    """
    Create a ToolNode with a fallback to handle errors and surface them to the agent.
    """
    return ToolNode(tools).with_fallbacks(
        [RunnableLambda(handle_tool_error)], exception_key="error"
    )


def handle_tool_error(state) -> dict:
    error = state.get("error")
    tool_calls = state["messages"][-1].tool_calls
    return {
        "messages": [
            ToolMessage(
                content=f"Error: {repr(error)}\n please fix your mistakes.",
                tool_call_id=tc["id"],
            )
            for tc in tool_calls
        ]
    }



# ----------------------------------------------------------------
# 4) Tools for the Agent
# ----------------------------------------------------------------
if st.session_state.get("db"):
    toolkit = SQLDatabaseToolkit(
        db=st.session_state["db"], 
        llm=AzureChatOpenAI(
            deployment_name=AZURE_OPENAI_DEPLOYMENT,
            model_name="gpt-4o",
            azure_endpoint=AZURE_OPENAI_ENDPOINT,
            azure_deployment=AZURE_OPENAI_DEPLOYMENT,
            api_version=AZURE_OPENAI_API_VERSION,
            api_key=AZURE_OPENAI_API_KEY,
            temperature=0.0,
        ),
    )
    tools = toolkit.get_tools()

    list_tables_tool = next(tool for tool in tools if tool.name == "sql_db_list_tables")
    get_schema_tool = next(tool for tool in tools if tool.name == "sql_db_schema")
    db_query_tool = next(tool for tool in tools if tool.name == "sql_db_query")

else:
    st.warning("Database is not initialized. Please upload a file to proceed.")


# ----------------------------------------------------------------
# 5) Define a small "State" for LangGraph
# ----------------------------------------------------------------
class State(TypedDict):
    messages: Annotated[list[AnyMessage], add_messages]


# ----------------------------------------------------------------
# 6) Query Checking Prompt
# ----------------------------------------------------------------
query_check_system = """You are a SQL expert with a strong attention to detail.
Double check the SQLite query for common mistakes, including:
- Using NOT IN with NULL values
- Using UNION when UNION ALL should have been used
- Using BETWEEN for exclusive ranges
- Data type mismatch in predicates
- Properly quoting identifiers
- Using the correct number of arguments for functions
- Casting to the correct data type
- Using the proper columns for joins

If there are any of the above mistakes, rewrite the query. If there are no mistakes, just reproduce the original query.

You will call the appropriate tool to execute the query after running this check.
"""

if st.session_state.get("db"):
    query_check_prompt = ChatPromptTemplate.from_messages(
        [("system", query_check_system), ("placeholder", "{messages}")]
    )

    query_check = query_check_prompt | AzureChatOpenAI(
            deployment_name=AZURE_OPENAI_DEPLOYMENT,
            model_name="gpt-4o",
            azure_endpoint=AZURE_OPENAI_ENDPOINT,
            azure_deployment=AZURE_OPENAI_DEPLOYMENT,
            api_version=AZURE_OPENAI_API_VERSION,
            api_key=AZURE_OPENAI_API_KEY,
            temperature=0.0,
        ).bind_tools(
        [db_query_tool]
    )

def model_check_query(state: State) -> dict[str, list[AIMessage]]:
    """
    Use this tool to double-check if your query is correct before executing it.
    """
    return {"messages": [query_check.invoke({"messages": [state["messages"][-1]]})]}


# ----------------------------------------------------------------
# 7) Query Generation Prompt
# ----------------------------------------------------------------
class SubmitFinalAnswer(BaseModel):
    """Submit the final answer to the user based on the query results."""
    final_answer: str = Field(..., description="The final answer to the user")

query_gen_system = """You are an advanced SQL expert with exceptional attention to detail and a focus on accuracy. Your role is to query an SQLite database to answer input questions by generating syntactically correct SQL queries and analyzing the results.

DO NOT call any tool besides SubmitFinalAnswer to submit the final answer.

When generating the query:

1. Output a syntactically correct SQLite query that answers the input question, considering only the relevant columns and tables in the database. Avoid querying all columns unless explicitly asked in the question.
2. Include all rows in the query result that are relevant to answer the user question. Identify which rows of the database contain the information asked for to facilitate user understanding of the answer.
3. If an error occurs while executing a query, rewrite the query and try again.
4. If the query returns an empty result set, rewrite it to retrieve a non-empty result set, provided it aligns with the input question. Never fabricate data or assumptions to fill gaps.
5. Do not make any Data Manipulation Language (DML) statements (e.g., INSERT, UPDATE, DELETE, DROP).

When returning the results:

1. Display all rows returned by the query and explicitly highlight the specific row or rows that are relevant to the question.
2. If discrepancies or calculations are requested, include clear explanations of the results alongside the highlighted rows.
3. If there isn't enough information in the database to answer the question, respond with: "The database does not contain sufficient information to answer this question."

Remember:
- Only submit a final answer to the user after validating the query results.
- Do not use any tool or invoke any external logic except `SubmitFinalAnswer` for the final output.
- Prioritize clarity and precision when presenting the results.

Your primary objectives are to ensure correctness in SQL query construction, exhaustiveness in result presentation, and relevance in highlighting the rows or calculations pertinent to the question.
"""

query_gen_prompt = ChatPromptTemplate.from_messages(
    [("system", query_gen_system), ("placeholder", "{messages}")]
)


query_gen = query_gen_prompt | AzureChatOpenAI(
            deployment_name=AZURE_OPENAI_DEPLOYMENT,
            model_name="gpt-4o",
            azure_endpoint=AZURE_OPENAI_ENDPOINT,
            azure_deployment=AZURE_OPENAI_DEPLOYMENT,
            api_version=AZURE_OPENAI_API_VERSION,
            api_key=AZURE_OPENAI_API_KEY,
            temperature=0.0,
        ).bind_tools(
    [SubmitFinalAnswer]
)


def query_gen_node(state: State):
    message = query_gen.invoke(state)

    # Sometimes, the LLM will hallucinate and call the wrong tool. We need to catch this and return an error message.
    tool_messages = []
    if message.tool_calls:
        for tc in message.tool_calls:
            if tc["name"] != "SubmitFinalAnswer":
                tool_messages.append(
                    ToolMessage(
                        content=f"Error: The wrong tool was called: {tc['name']}. Please fix your mistakes. Remember to only call SubmitFinalAnswer to submit the final answer. Generated queries should be outputted WITHOUT a tool call.",
                        tool_call_id=tc["id"],
                    )
                )
    else:
        tool_messages = []
    return {"messages": [message] + tool_messages}


# ----------------------------------------------------------------
# 8) Build the Workflow
# ----------------------------------------------------------------
def first_tool_call(state: State) -> dict[str, list[AIMessage]]:
    return {
        "messages": [
            AIMessage(
                content="",
                tool_calls=[
                    {
                        "name": "sql_db_list_tables",
                        "args": {},
                        "id": "tool_abcd123",
                    }
                ],
            )
        ]
    }


# Define a conditional edge to decide whether to continue or end the workflow
def should_continue(state: State) -> Literal[END, "correct_query", "query_gen"]: # type: ignore
    messages = state["messages"]
    last_message = messages[-1]
    # If there is a tool call, then we finish
    if getattr(last_message, "tool_calls", None):
        return END
    if last_message.content.startswith("Error:"):
        return "query_gen"
    else:
        return "correct_query"

def build_workflow():
    workflow = StateGraph(State)

    # Node definitions
    workflow.add_node("first_tool_call", first_tool_call)
    workflow.add_node("list_tables_tool", create_tool_node_with_fallback([list_tables_tool]))
    workflow.add_node("get_schema_tool", create_tool_node_with_fallback([get_schema_tool]))
    
    workflow.add_node(
        "model_get_schema",
        lambda state: {
            "messages": [
                AzureChatOpenAI(
                    deployment_name=AZURE_OPENAI_DEPLOYMENT,
                    model_name="gpt-4o",
                    azure_endpoint=AZURE_OPENAI_ENDPOINT,
                    azure_deployment=AZURE_OPENAI_DEPLOYMENT,
                    api_version=AZURE_OPENAI_API_VERSION,
                    api_key=AZURE_OPENAI_API_KEY,
                    temperature=0.0,
                ).bind_tools([get_schema_tool]).invoke(state["messages"])
            ]
        },
    )
    workflow.add_node("query_gen", query_gen_node)
    workflow.add_node("correct_query", model_check_query)
    workflow.add_node("execute_query", create_tool_node_with_fallback([db_query_tool]))

    # Edges
    workflow.add_edge(START, "first_tool_call")
    workflow.add_edge("first_tool_call", "list_tables_tool")
    workflow.add_edge("list_tables_tool", "model_get_schema")
    workflow.add_edge("model_get_schema", "get_schema_tool")
    workflow.add_edge("get_schema_tool", "query_gen")
    workflow.add_conditional_edges("query_gen", should_continue)
    workflow.add_edge("correct_query", "execute_query")
    workflow.add_edge("execute_query", "query_gen")

    return workflow.compile()




# ----------------------------------------------------------------
# 9) The Streamlit App
# ----------------------------------------------------------------
def main():
    st.title("Q&A over your recon data")

    if "workflow" not in st.session_state:
        st.session_state["workflow"] = None

    if "db" not in st.session_state:
        st.session_state["db"] = None

    # 1) Upload
    uploaded_file = st.file_uploader("Upload CSV or SQLite DB (.db/.sqlite)", type=["csv", "db", "sqlite"])

    if uploaded_file is not None:
        # 2) If CSV, convert it to a file-based DB
        if uploaded_file.name.lower().endswith(".csv"):
            db_path = load_csv_to_file_db(uploaded_file)
            if db_path:
                st.session_state["db"] = build_sql_database_from_path(db_path)
                st.success("CSV loaded into file-based SQLite database (temp_uploaded_db.sqlite).")
        # 3) If .db/.sqlite, save as local DB
        elif uploaded_file.name.lower().endswith(".db") or uploaded_file.name.lower().endswith(".sqlite"):
            db_path = load_db_file(uploaded_file)
            if db_path:
                st.session_state["db"] = build_sql_database_from_path(db_path)
                st.success("SQLite database loaded.")
        else:
            st.error("Unsupported file type.")

        if st.session_state["db"] is not None:
            # 4) Build the toolkit & tools
            toolkit = SQLDatabaseToolkit(
                db=st.session_state["db"],
                llm=AzureChatOpenAI(
                    deployment_name=AZURE_OPENAI_DEPLOYMENT,
                    model_name="gpt-4o",
                    azure_endpoint=AZURE_OPENAI_ENDPOINT,
                    azure_deployment=AZURE_OPENAI_DEPLOYMENT,
                    api_version=AZURE_OPENAI_API_VERSION,
                    api_key=AZURE_OPENAI_API_KEY,
                    temperature=0.0,
                ),
            )
            all_tools = toolkit.get_tools()

            global list_tables_tool, get_schema_tool, db_query_tool, tool_choices
            list_tables_tool = next(t for t in all_tools if t.name == "sql_db_list_tables")
            get_schema_tool = next(t for t in all_tools if t.name == "sql_db_schema")
            db_query_tool = next(tool for tool in all_tools if tool.name == "sql_db_query")


            # Our "tool_choices" for query_check step
            tool_choices = [db_query_tool, list_tables_tool, get_schema_tool]

            # Rebuild the workflow
            st.session_state["workflow"] = build_workflow()

    user_question = st.text_input("Ask a question about your data:")

    # 5) Run the workflow
    if st.button("Run Query") and user_question.strip():
        if not st.session_state["workflow"]:
            st.warning("Please upload a file first.")
            return

        input_data = {"messages": [("user", user_question)]}

        st.write("### Workflow execution:")
        with st.expander("Show step-by-step events"):
            for event in st.session_state["workflow"].stream(input_data):
                st.write(event)

        output = st.session_state["workflow"].invoke(input_data)
        final_msgs = output.get("messages", [])
        if not final_msgs:
            st.warning("No final messages returned.")
            return

        last_msg = final_msgs[-1]
        st.write("### Final Output:")
        # If the last message has a tool call to SubmitFinalAnswer
        if getattr(last_msg, "tool_calls", None):
            if "final_answer" in last_msg.tool_calls[0].get("args", {}):
                st.success(last_msg.tool_calls[0]["args"]["final_answer"])
            else:
                st.warning("No 'final_answer' in the last message tool call.")
        else:
            # Just textual content
            st.success(last_msg.content)

if __name__ == "__main__":
    main()
