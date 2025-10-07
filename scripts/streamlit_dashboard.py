import os
import json
import streamlit as st

# Paths
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUTPUT_PATH = os.path.join(BASE_DIR, "outputs", "evaluated_summaries.json")

# Load evaluated summaries
with open(OUTPUT_PATH, "r", encoding="utf-8") as f:
    data = json.load(f)

st.title("Meeting Summaries Evaluation Dashboard")

# Sidebar to select a meeting
meeting_ids = [item["meeting_id"] for item in data]
selected_id = st.sidebar.selectbox("Select Meeting ID", meeting_ids)

# Get selected meeting data
meeting = next(item for item in data if item["meeting_id"] == selected_id)

st.header(f"Meeting ID: {selected_id}")

st.subheader("Summary Metrics")
st.write(f"Reference Length: {meeting['ref_len']} words")
st.write(f"Generated Length: {meeting['gen_len']} words")
st.write(f"Length Ratio (gen/ref): {meeting['length_ratio']:.2f}")
st.write(f"ROUGE-1 F1: {meeting['rouge1_f1']:.3f}")
st.write(f"ROUGE-L F1: {meeting['rougeL_f1']:.3f}")

st.subheader("Keywords")
st.write(f"Reference Keywords: {meeting['ref_keywords']}")
st.write(f"Generated Keywords: {meeting['gen_keywords']}")
st.write(f"Missing Keywords: {meeting['missing_keywords']}")

st.subheader("Generated Summary")
st.text_area("Generated Summary", value=meeting.get("generated_summary", ""), height=150)

st.subheader("Reference Summary")
st.text_area("Reference Summary", value=meeting.get("reference", ""), height=150)
