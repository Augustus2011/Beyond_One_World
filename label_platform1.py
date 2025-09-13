import streamlit as st
import pandas as pd
from openai import OpenAI

st.set_page_config(page_title="CSV Annotation Tool", page_icon="📑", layout="wide")

st.title("📑 CSV Annotation Platform")
uploaded_file = st.file_uploader("Upload a CSV file", type=["csv"])
if uploaded_file:
    df = pd.read_csv(uploaded_file)
    

    if 'annotation' not in df.columns:
        df['annotation'] = ""

    edited_df = st.data_editor(df, num_rows="dynamic")

    saved_name = st.text_input("Enter file name to save, e.g., new_character_data_august.csv")
    if st.button("Save CSV"):
        edited_df.to_csv(saved_name, index=False)
        st.success(f"CSV saved successfully as '{saved_name}'!")


api_key = st.text_input("Enter your OpenAI API(4o only) Key", type="password")

client = OpenAI(api_key=api_key) if api_key else None

st.divider()
st.subheader("💬 AI Annotation Assistant")


st.markdown("""
            **Prompt Templates for canon Q&A:**

```
your job is to generate question and answer about decision of that situation \n
information: "<key0 name from source>,<key1 when>,<key2 where>,<key3 what is happening>,<key4 question>" \n
the decision: "<answer>" \n
generate question from information and choices of decision A) B) C) D),one for the answer other are not
""")


st.markdown("""
            **Prompt Templates for dilemma Q:**

```
your job is to generate dilemma situation \n
information: "<key0 name from source>,<key1 when>,<key2 where>,<key3 what is happening>,<key4 dilemma situation>" \n
generate situation about that dilemma in detail, the situation must contains scene and dilemma two options and their Consequences
""")

if "messages" not in st.session_state:
    st.session_state.messages = []


for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

user_input = st.chat_input("Ask ChatGPT for annotation help...")

if user_input and client:
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)
    
    try:
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": "You are a helpful assistant"},
                {"role": "user", "content": user_input}
            ],
        )
        bot_response = response.choices[0].message.content

        st.session_state.messages.append({"role": "assistant", "content": bot_response})
        with st.chat_message("assistant"):
            st.markdown(bot_response)
    except Exception as e:
        st.error(f"Error: {e}")
