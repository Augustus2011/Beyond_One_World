
import streamlit as st
import pandas as pd
import json
import glob
import os

st.set_page_config(page_title="Annotation Tool", page_icon="📑", layout="wide")

uploaded_files = st.sidebar.file_uploader("Upload JSONL files", type=["jsonl"], accept_multiple_files=True)

merged_file = "merged_annotations.jsonl"
if uploaded_files:
    all_data = []
    for uploaded_file in uploaded_files:
        content = uploaded_file.getvalue().decode("utf-8").splitlines()
        for line in content:
            record = json.loads(line)
            record["source_file"] = uploaded_file.name
            all_data.append(record)
    
    with open(merged_file, "w", encoding="utf-8") as f:
        for record in all_data:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
    st.sidebar.success("Files merged successfully!")


jsonl_files = glob.glob("*.jsonl")
st.sidebar.title("📂 Select Annotation File")
selected_file = st.sidebar.selectbox("Choose a JSONL file", jsonl_files)

if selected_file:
    with open(selected_file, "r", encoding="utf-8") as f:
        data = [json.loads(line) for line in f]
    
    df = pd.DataFrame(data)
    if "thinking_score" not in df.columns:
        df["thinking_score"] = None
    if "acting_score" not in df.columns:
        df["acting_score"] = None
    if "describe_score" not in df.columns:
        df["describe_score"] = None

    unique_cids = df["CID"].unique()

    # Session state for navigation
    if "current_index" not in st.session_state:
        st.session_state.current_index = 0
    
    current_cid = unique_cids[st.session_state.current_index]
    cid_data = df[df["CID"] == current_cid]
    rows = cid_data.to_dict(orient="records")
    
    if "row_index" not in st.session_state:
        st.session_state.row_index = 0
    
    st.title(f"📑 Annotation - {selected_file}")
    st.subheader(f"Annotating CID: {current_cid}")

    if rows:
        row = rows[st.session_state.row_index]
        
        col1, col2 = st.columns([1, 1])
        with col1:
            if st.button("⬅️ Previous Row") and st.session_state.row_index > 0:
                st.session_state.row_index -= 1
                st.rerun()
        with col2:
            if st.button("➡️ Next Row") and st.session_state.row_index < len(rows) - 1:
                st.session_state.row_index += 1
                st.rerun()

        with st.expander(f"Question for {row['name']}"):
            st.text_area("Question:", row.get("question", "No question available"), height=400, disabled=True)
        with st.expander(f"Answer for {row['name']}"):
            st.text_area("Answer:", row.get("answers", "No answer available"), height=400, disabled=True)
    
    col3, col4 = st.columns([1, 1])
    with col3:
        if st.button("⬅️ Previous CID") and st.session_state.current_index > 0:
            st.session_state.current_index -= 1
            st.session_state.row_index = 0
            st.rerun()
    with col4:
        if st.button("➡️ Next CID") and st.session_state.current_index < len(unique_cids) - 1:
            st.session_state.current_index += 1
            st.session_state.row_index = 0
            st.rerun()
    

    st.subheader("📋 Annotation Table")
    editable_df = st.data_editor(cid_data, num_rows="dynamic")
    
    save_filename = st.text_input("annotated_data.jsonl")
    if st.button("Save Table Annotations"):
        df.update(editable_df)
        with open(save_filename, "w", encoding="utf-8") as f:
            for record in df.to_dict(orient="records"):
                f.write(json.dumps(record, ensure_ascii=False) + "\n")
        st.success(f"Table annotations saved as {save_filename}!")
