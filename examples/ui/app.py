import os
import requests
import streamlit as st
from dotenv import load_dotenv

# Load environment variables
load_dotenv()
api_host = os.environ.get("HOST", "0.0.0.0")
api_port = int(os.environ.get("PORT", 8080))

# Streamlit UI elements
st.title("📈 Nifty 50 Stock Insights")
st.markdown(
    """
    ## How to use:
    
    Enter a question about any of the Nifty 50 stocks, and the AI will provide insights based on the latest stock data.

    ---
    """
)

question = st.text_input(
    "Enter your question here",
    placeholder="E.g., What is the current trend for Reliance Industries?",
)

# Handle the query submission
if question:
    url = f'http://{api_host}:{api_port}/'
    data = {"query": question}

    response = requests.post(url, json=data)

    if response.status_code == 200:
        st.write("### Answer")
        st.write(response.json())
    else:
        st.error(f"Failed to obtain insights. Status code: {response.status_code}")
