from agent import agent
import streamlit as st
from langchain_groq import ChatGroq
import os

# Initialize Groq LLM
llm = ChatGroq(
    model="llama-3.1-8b-instant",
    temperature=0
)

st.set_page_config(page_title="AI CRM Demo", layout="wide")
st.title("AI-First CRM – HCP Interaction Demo")

hcp_data = {
    "Dr. Smith": "Discussed PainRelief Plus dosage and effectiveness.",
    "Dr. Johnson": "Concerned about HeartSafe side effects.",
    "Dr. Lee": "Asked about AllergyX clinical trials."
}

selected_hcp = st.selectbox("Select HCP", list(hcp_data.keys()))
interaction_text = st.text_area(
    "HCP Interaction",
    hcp_data[selected_hcp],
    height=150
)

if st.button("Analyze Interaction"):
    with st.spinner("Analyzing with LangGraph Agent..."):
        result = agent.invoke({
            "input_text": interaction_text
        })

    st.success("Analysis Complete")

    st.subheader("📌 Summary")
    st.write(result["summary"])

    st.subheader("😊 Sentiment")
    st.write(result["sentiment"])

    st.subheader("⚠️ Compliance Risk")
    st.write(result["compliance"])

    st.subheader("➡️ Next Best Action")
    st.write(result["next_action"])
