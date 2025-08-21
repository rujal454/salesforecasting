import streamlit as st
import pandas as pd
import requests
import os

API_BASE = os.getenv("API_BASE", "http://127.0.0.1:8000")

st.title("📊 Sales Forecasting AI + RAG Chatbot")

st.sidebar.header("Upload & Index Data")
uploaded = st.sidebar.file_uploader("Upload CSV", type="csv")
text_col = st.sidebar.text_input("Text column for embeddings", value="product")

if uploaded and st.sidebar.button("Index to Chroma"):
    files = {"csv": uploaded}
    data = {"text_column": text_col}
    try:
        r = requests.post(f"{API_BASE}/embed-index", files=files, data=data)
        r.raise_for_status()
        st.sidebar.success(f"Indexed: {r.json().get('indexed', 0)} docs")
    except Exception as e:
        st.sidebar.error(f"Indexing failed: {e}")
        st.sidebar.write("Raw response:", getattr(r, "text", ""))

st.header("Sales Forecast")
uploaded2 = st.file_uploader("Upload CSV for Forecast", type="csv", key="forecast")
if uploaded2 and st.button("Forecast"):
    files = {"csv": uploaded2}
    try:
        r = requests.post(f"{API_BASE}/forecast", files=files)
        r.raise_for_status()
        resp_json = r.json()
        if "forecast" in resp_json:
            forecast = pd.DataFrame(resp_json["forecast"])
            st.line_chart(forecast.set_index("date")["forecast"])
        else:
            st.error("No forecast data returned.")
            st.write("Raw response:", resp_json)
    except Exception as e:
        st.error(f"Forecast failed: {e}")
        try:
            st.write("Raw response:", getattr(r, "text", ""))
        except:
            st.write("No response available")

st.header("RAG Chatbot")
query = st.text_input("Ask a question about your sales data:")
if st.button("Ask"):
    if query.strip():
        with st.spinner("🤖 AI is thinking..."):
            try:
                r = requests.post(f"{API_BASE}/chat", data={"query": query})
                r.raise_for_status()
                resp_json = r.json()

                if "answer" in resp_json and resp_json["answer"]:
                    st.success("✅ AI Response:")
                    st.write(resp_json["answer"])

                    # Show additional info if available
                    if "model" in resp_json:
                        st.info(f"Model: {resp_json['model']}")
                    if "tokens_used" in resp_json:
                        st.info(f"Tokens used: {resp_json['tokens_used']}")

                elif "error" in resp_json:
                    st.error(f"❌ Error: {resp_json['error']}")
                    if "raw_response" in resp_json:
                        st.write("Raw response:", resp_json["raw_response"])
                else:
                    st.warning("⚠️ No answer returned from AI")
                    st.write("Full response:", resp_json)

            except Exception as e:
                st.error(f"❌ Chat failed: {e}")
                try:
                    st.write("Raw response:", r.text if 'r' in locals() else "No response")
                except:
                    st.write("No response available")
    else:
        st.warning("Please enter a question.")