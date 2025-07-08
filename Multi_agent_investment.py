from langchain_ollama import ChatOllama
from langchain_core.messages import ToolMessage, HumanMessage, convert_to_messages
from langchain_core.tools import tool
from langgraph.graph import MessagesState, StateGraph
from langgraph.types import Command
from typing_extensions import Literal
from IPython.display import Image, display

@tool
def ask_tax_advisor():
    """Use this tool to transfer the question to the Tax Advisor if investment advice needs tax clarification."""
    return "Passing to Tax Advisor"

@tool
def ask_finance_advisor():
    """Use this tool to transfer the question to the Finance Advisor if tax advice needs investment insight."""
    return "Passing to Finance Advisor"

model = ChatOllama(model="llama3.2")


def finance_advisor(state: MessagesState) -> Command[Literal["tax_advisor", "__end__"]]:
    system_prompt = (
        "You are a Finance Advisor. Recommend investment strategies. "
        "If tax concerns arise, use the 'ask_tax_advisor' tool."
    )

    messages = [{"role": "system", "content": system_prompt}] + state["messages"]
    ai_msg = model.bind_tools([ask_tax_advisor]).invoke(messages)

    if ai_msg.tool_calls:
        tool_call_id = ai_msg.tool_calls[-1]["id"]
        tool_msg = {
            "role": "tool",
            "content": "Successfully transferred",
            "tool_call_id": tool_call_id,
        }
        return Command(goto="tax_advisor", update={"messages": [ai_msg, tool_msg]})

    return {"messages": [ai_msg]}


def tax_advisor(state: MessagesState) -> Command[Literal["finance_advisor", "__end__"]]:
    system_prompt = (
        "You are a Tax Advisor. Provide tax advice. "
        "If investment guidance is needed, use the 'ask_finance_advisor' tool."
    )

    messages = [{"role": "system", "content": system_prompt}] + state["messages"]
    ai_msg = model.bind_tools([ask_finance_advisor]).invoke(messages)

    if ai_msg.tool_calls:
        tool_call_id = ai_msg.tool_calls[-1]["id"]
        tool_msg = {
            "role": "tool",
            "content": "Successfully transferred",
            "tool_call_id": tool_call_id,
        }
        return Command(goto="finance_advisor", update={"messages": [ai_msg, tool_msg]})

    return {"messages": [ai_msg]}


START = "finance_advisor"
graph = StateGraph(MessagesState)
graph.set_entry_point(START)
graph.add_node("finance_advisor", finance_advisor)
graph.add_node("tax_advisor", tax_advisor)
app = graph.compile()


def show_graph():
    display(Image(app.get_graph().draw_mermaid_png()))


def evaluate_vs_raw_model(query: str):
    print("Running MAS (LangGraph) agent...\n")
    mas_response_chunks = []
    for chunk in app.stream({"messages": [("user", query)]}):
        mas_response_chunks.append(chunk)

    final_messages = []
    for update in mas_response_chunks:
        for node_update in update.values():
            messages = convert_to_messages(node_update["messages"])
            for m in messages:
                if m.type == "ai":
                    final_messages.append(m.content)

    mas_output = "\n".join(final_messages)
    print("MAS Agent Response:\n", mas_output)

    print("\n" + "=" * 80 + "\n")

    print("Running raw llama3.2 model (direct)...\n")
    eval_model = ChatOllama(model="llama3.2")
    raw_response = eval_model.invoke([HumanMessage(content=query)])
    raw_output = raw_response.content
    print("Direct Model Response:\n", raw_output)

    return mas_output, raw_output


def judge_responses(user_query, mas_output, raw_output):
    judge_prompt = f"""
You are a helpful evaluator. Compare the two responses below to the user question.

Question:
{user_query}

Response from MAS Agent:
{mas_output}

Response from Raw Model:
{raw_output}

Which one provides clearer, more helpful, and contextually appropriate advice? Reply with "MAS", "Raw", or "Tie", and explain your reasoning.
"""
    judge = ChatOllama(model="llama3.2")
    judgment = judge.invoke([HumanMessage(content=judge_prompt)])
    print("\nJudge Response:\n")
    print(judgment.content)


def run_demo():
    query = "suggest how to invest money"
    mas_output, raw_output = evaluate_vs_raw_model(query)
    judge_responses(query, mas_output, raw_output)

if __name__ == "__main__":
    run_demo()
