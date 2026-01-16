from typing import TypedDict
from langgraph.graph import StateGraph
from langchain_groq import ChatGroq
import os

# ---- LLM ----
llm = ChatGroq(
    api_key=os.environ.get("GROQ_API_KEY"),
    model="llama-3.1-8b-instant",
    temperature=0
)

# ---- STATE ----
class InteractionState(TypedDict):
    input_text: str
    summary: str
    sentiment: str
    compliance: str
    next_action: str

# ---- TOOLS ----

def log_interaction(state: InteractionState):
    response = llm.invoke(
        f"Summarize this HCP interaction:\n{state['input_text']}"
    )
    return {"summary": response.content}

def analyze_sentiment(state: InteractionState):
    response = llm.invoke(
        f"Classify sentiment (Positive, Neutral, Negative):\n{state['input_text']}"
    )
    return {"sentiment": response.content}

def compliance_check(state: InteractionState):
    response = llm.invoke(
        f"""
        Check if this interaction has compliance risk
        in life sciences sales. Answer Yes or No.
        Interaction:
        {state['input_text']}
        """
    )
    return {"compliance": response.content}

def next_best_action(state: InteractionState):
    response = llm.invoke(
        f"Suggest next best sales action for this HCP interaction:\n{state['input_text']}"
    )
    return {"next_action": response.content}

def edit_interaction(state: InteractionState):
    return state  # placeholder for manual edits

# ---- LANGGRAPH ----

graph = StateGraph(InteractionState)

graph.add_node("log", log_interaction)
graph.add_node("sentiment", analyze_sentiment)
graph.add_node("compliance", compliance_check)
graph.add_node("next_action", next_best_action)

graph.set_entry_point("log")
graph.add_edge("log", "sentiment")
graph.add_edge("sentiment", "compliance")
graph.add_edge("compliance", "next_action")

agent = graph.compile()
