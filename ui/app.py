
import requests
import streamlit as st

st.set_page_config(
    page_title="AI Investigation Engine",
    page_icon="🔎",
    layout="wide",
)

st.title("AI Investigation Engine")
st.write("Ask questions and investigate evidence across documents.")

api_url = st.sidebar.text_input(
    "Backend URL",
    value="http://127.0.0.1:8000",
)

question = st.text_area(
    "Investigation question",
    placeholder="What happened, and which documents support the conclusion?",
)

if st.button("Investigate", type="primary"):
    if not question.strip():
        st.warning("Enter a question first.")
    else:
        try:
            response = requests.post(
                f"{api_url.rstrip('/')}/ask",
                json={"question": question},
                timeout=30,
            )
            response.raise_for_status()
            result = response.json()

            st.subheader("Result")
            if result.get("answer"):
                st.write(result["answer"])
            else:
                st.info(result.get("message", "No supported answer yet."))

            st.write("Abstained:", result.get("abstained", True))
            st.write("Citations:", result.get("citations", []))

        except requests.RequestException as exc:
            st.error(f"Could not reach the backend: {exc}")
