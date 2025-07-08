import os
import streamlit as st
from dotenv import load_dotenv
from langchain_cohere import ChatCohere
from langchain_core.messages import ToolMessage, HumanMessage, convert_to_messages
from langchain_core.tools import tool
from langgraph.graph import MessagesState, StateGraph
from langgraph.types import Command
from typing_extensions import Literal
from tenacity import retry, stop_after_attempt, wait_exponential
# Load environment variable
load_dotenv()
api_key = os.getenv("COHERE_API_KEY")

# Define tools
@tool
def ask_tax_advisor():
    """Use this tool to transfer the question to the Tax Advisor if investment advice needs tax clarification."""
    return "Passing to Tax Advisor"

@tool
def ask_finance_advisor():
    """Use this tool to transfer the question to the Finance Advisor if tax advice needs investment insight."""
    return "Passing to Finance Advisor"

# Set up model
model = ChatCohere(model="command-r-plus", cohere_api_key=api_key)

# Define agents
def finance_advisor(state: MessagesState) -> Command[Literal["tax_advisor", "__end__"]]:
    messages = [{"role": "system", "content": "You are a Finance Advisor. Recommend detailed and wide explanation of the investment strategies. "
                                              "If tax concerns arise, use the 'ask_tax_advisor' tool."
                                              "Provide a detailed-wide explananation of the query asked by the user better than the raw model."}] + state["messages"]
    ai_msg = model.bind_tools([ask_tax_advisor]).invoke(messages)

    if ai_msg.tool_calls:
        tool_msg = ToolMessage(
            tool_call_id=ai_msg.tool_calls[-1]["id"],
            content="Passing to Tax Advisor"
        )
        return Command(goto="tax_advisor", update={"messages": [ai_msg, tool_msg]})

    return {"messages": [ai_msg]}


def tax_advisor(state: MessagesState) -> Command[Literal["finance_advisor", "__end__"]]:
    messages = [{"role": "system", "content": "You are a Tax Advisor. Provide tax advice. "
                                              "If investment guidance is needed, use the 'ask_finance_advisor' tool."}] + state["messages"]
    ai_msg = model.bind_tools([ask_finance_advisor]).invoke(messages)

    if ai_msg.tool_calls:
        tool_msg = ToolMessage(
            tool_call_id=ai_msg.tool_calls[-1]["id"],
            content="Passing to Finance Advisor"
        )
        return Command(goto="finance_advisor", update={"messages": [ai_msg, tool_msg]})

    return {"messages": [ai_msg]}

# Build LangGraph
graph = StateGraph(MessagesState)
graph.set_entry_point("finance_advisor")
graph.add_node("finance_advisor", finance_advisor)
graph.add_node("tax_advisor", tax_advisor)
app = graph.compile()

# Define helper functions


@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=10))
def run_mas_system(user_query):
    mas_chunks = []
    for chunk in app.stream({"messages": [("user", user_query)]}):
        mas_chunks.append(chunk)
    responses = []
    for update in mas_chunks:
        for node_update in update.values():
            for m in convert_to_messages(node_update["messages"]):
                if m.type == "ai":
                    responses.append(m.content)
    return "\n".join(responses)

def run_raw_model(user_query):
    eval_model = ChatCohere(model="command-r-plus", cohere_api_key=api_key)
    response = eval_model.invoke([HumanMessage(content=user_query)])
    return response.content

def judge(user_query, mas_output, raw_output):
    prompt = f"""
You are a helpful evaluator. Compare the two responses below to the user question.

Question:
{user_query}

Response from MAS Agent:
{mas_output}

Response from Raw Model:
{raw_output}

Which one provides clearer, more helpful, and contextually appropriate advice? Reply with "MAS", "Raw", or "Tie", and explain your reasoning.
"""
    judge_model = ChatCohere(model="command-r-plus", cohere_api_key=api_key)
    result = judge_model.invoke([HumanMessage(content=prompt)])
    return result.content

# Build Streamlit UI
st.set_page_config(page_title="Multi-Agent System vs Raw Model Evaluation", layout="centered")

st.title("📊 Multi-Agent System vs Raw Model Chat Evaluation")
st.write("Enter your financial or tax-related question below.")

query = st.text_input("💬 Your Question", placeholder="e.g., How can I reduce my taxes while investing?")
if st.button("Evaluate"):
    if not query.strip():
        st.warning("Please enter a question.")
    else:
        with st.spinner("Running MAS system..."):
            mas_output = run_mas_system(query)
        st.success("MAS Agent response generated.")

        with st.spinner("Running raw model..."):
            raw_output = run_raw_model(query)
        st.success("Raw model response generated.")

        st.subheader("📥 MAS Agent Response")
        st.write(mas_output)

        st.subheader("📤 Raw Model Response")
        st.write(raw_output)

        with st.spinner("Judging..."):
            verdict = judge(query, mas_output, raw_output)
        st.subheader("🧑‍⚖️ Evaluation Verdict")
        st.write(verdict)
