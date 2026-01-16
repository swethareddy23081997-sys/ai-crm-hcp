# frontend.py

import streamlit as st
from agent import agent  # LangGraph agent
from datetime import datetime
import os
from langchain_groq import ChatGroq

# Page settings
st.set_page_config(page_title="AI-First CRM – HCP Interaction", layout="wide")
st.title("Log HCP Interaction")

# Sample HCP list (replace with database or API later)
hcp_list = ["Dr. Smith", "Dr. Johnson", "Dr. Lee"]

# Layout: Left = Interaction Form, Right = AI Assistant
left_col, right_col = st.columns([3, 1])

with left_col:
    st.subheader("Interaction Details")
    
    # HCP Name
    hcp_name = st.selectbox("HCP Name", hcp_list)
    
    # Interaction Type
    interaction_type = st.selectbox("Interaction Type", ["Meeting", "Call", "Email"])
    
    # Date & Time
    date = st.date_input("Date", datetime.today())
    time = st.time_input("Time", datetime.now().time())
    
    # Attendees
    attendees = st.text_input("Attendees (names)")
    
    # Topics Discussed
    topics = st.text_area("Topics Discussed", height=120)
    
    # Materials Shared / Samples
    materials_shared = st.text_area("Materials Shared / Samples Distributed")
    
    # Observed/Infered Sentiment
    sentiment = st.radio(
        "Observed / Inferred HCP Sentiment",
        ["Positive", "Neutral", "Negative"]
    )
    
    # Outcomes
    outcomes = st.text_area("Outcomes / Agreements", height=100)
    
    # Follow-up Actions
    follow_up = st.text_area("Follow-up Actions", height=100)

with right_col:
    st.subheader("AI Assistant")
    interaction_input = st.text_area(
        "Describe interaction (AI will summarize and suggest next steps)"
    )
    if st.button("Log Interaction"):
        if not interaction_input.strip():
            st.warning("Please enter some interaction text for AI analysis.")
        else:
            with st.spinner("Analyzing with LangGraph AI..."):
                # Call LangGraph agent
                result = agent.invoke({"input_text": interaction_input})

            # Display results
            st.markdown("### 📌 Summary")
            st.write(result.get("summary", "No summary generated."))

            st.markdown("### 😊 Sentiment")
            st.write(result.get("sentiment", "No sentiment detected."))

            st.markdown("### ⚠️ Compliance Risk")
            st.write(result.get("compliance", "Not assessed."))

            st.markdown("### ➡️ Next Best Action")
            next_actions = result.get("next_action", [])
            if isinstance(next_actions, list):
                for action in next_actions:
                    st.markdown(f"- {action}")
            else:
                st.write(next_actions)
