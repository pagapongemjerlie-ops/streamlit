import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from snowflake_connect import load_reviews 
from snowflake.snowpark import Session
import google.generativeai as genai


genai.configure(api_key="AIzaSyA1uDEXoMYCCsIn06dGXPArgv08HV-wHyU")
MODEL_NAME = "models/gemini-2.5-pro"
gemini_model = genai.GenerativeModel(MODEL_NAME)


st.set_page_config(page_title="Snowflake Sentiment Dashboard + Gemini Chatbot", layout="wide")


df = load_reviews()


df.columns = df.columns.str.strip().str.upper()


required_columns = {
    "PRODUCT": "Unknown Product",
    "CARRIER": np.random.choice(["UPS", "FedEx", "DHL"], size=len(df)),
    "SHIPPING_DATE": pd.to_datetime("2025-11-01") + pd.to_timedelta(np.random.randint(0, 30, size=len(df)), unit="D"),
    "STATUS": np.random.choice(["Delivered", "In Transit", "Pending"], size=len(df)),
    "LATE": np.random.choice([True, False], size=len(df)),
    "REGION": np.random.choice(["North", "South", "East", "West"], size=len(df)),
    "REVIEW_TEXT": "No review text available",
}

for col, default in required_columns.items():
    if col not in df.columns:
        df[col] = default
    elif col == "REVIEW_TEXT":
        
        df[col] = df[col].fillna("No review text available")


if "SENTIMENT_SCORE" not in df.columns:
    st.error("No SENTIMENT_SCORE column found!")
    st.stop()

df["SENTIMENT_SCORE"] = pd.to_numeric(df["SENTIMENT_SCORE"], errors="coerce")


st.sidebar.header("Filters")
products = st.sidebar.multiselect(
    "Select Product",
    df["PRODUCT"].unique(),
    default=df["PRODUCT"].unique()
)
filtered_df = df[df["PRODUCT"].isin(products)]


st.subheader("📄 Data Preview")
st.dataframe(filtered_df.head())


st.subheader("🌍 Average Sentiment Score by Product")
region_sentiment = (
    filtered_df.groupby("PRODUCT")["SENTIMENT_SCORE"].mean().sort_values()
).dropna()

fig, ax = plt.subplots()
region_sentiment.plot(kind="barh", ax=ax)
ax.set_ylabel("Product")
ax.set_xlabel("Average Sentiment")
ax.set_title("Sentiment by Product")
st.pyplot(fig)

st.write("### ❗ Products with Most Negative Sentiment")
st.write(region_sentiment.head())


st.subheader("🚚 Delivery Issues (Negative Sentiment)")
issues = filtered_df[(filtered_df["SENTIMENT_SCORE"] < 0) & (filtered_df["LATE"] == True)]

# Select columns safely
display_cols = ["PRODUCT", "REGION", "LATE", "SENTIMENT_SCORE", "REVIEW_TEXT"]
existing_cols = [c for c in display_cols if c in issues.columns]
issue_table = issues[existing_cols]
st.dataframe(issue_table)


st.subheader("🚛 Total Shipments by Carrier")
if "ORDER_ID" in filtered_df.columns:
    carrier_counts = filtered_df.groupby("CARRIER")["ORDER_ID"].count()
else:
    
    carrier_counts = filtered_df.groupby("CARRIER").size()

fig2, ax2 = plt.subplots()
carrier_counts.plot(kind="bar", ax=ax2)
ax2.set_ylabel("Number of Shipments")
ax2.set_title("Shipments by Carrier")
st.pyplot(fig2)


st.subheader("📊 Average Sentiment by Shipping Status")
status_sentiment = (
    filtered_df.groupby("STATUS")["SENTIMENT_SCORE"].mean().sort_values()
).dropna()

fig3, ax3 = plt.subplots()
status_sentiment.plot(kind="barh", ax=ax3)
ax3.set_xlabel("Sentiment Score")
ax3.set_ylabel("Shipping Status")
ax3.set_title("Avg Sentiment by Status")
st.pyplot(fig3)


from snowflake.snowpark import Session

def get_snowpark_session():
    connection_parameters = {
        "account": "IHIHVSG-YIC74668",  
        "user": "FUCK",
        "password": "Pikitpagapong174511",
        "role": "ACCOUNTADMIN",
        "warehouse": "COMPUTE_WH"
    }
    return Session.builder.configs(connection_parameters).create()

session = get_snowpark_session()



st.subheader("🤖 Chatbot Assistant")
chatbot_option = st.radio("Select Chatbot", ["Cortex LLM", "Gemini AI"])

if chatbot_option == "Cortex LLM":
    user_prompt = st.text_input("Ask about product reviews, trends, or sentiment:")
    if st.button("Send to Cortex"):
        if user_prompt.strip() != "":
            query = f"""
                SELECT snowflake.cortex.complete(
                    'llama3-70b',
                    'You are an AI assistant analyzing product reviews: {user_prompt}'
                ) AS answer;
            """
            result = session.sql(query).collect()[0]["ANSWER"]
            st.write(result)

elif chatbot_option == "Gemini AI":
    if "messages" not in st.session_state:
        st.session_state["messages"] = []

    
    for msg in st.session_state["messages"]:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    
    if prompt := st.chat_input("Type your message for Gemini AI..."):
        st.chat_message("user").markdown(prompt)
        st.session_state["messages"].append({"role": "user", "content": prompt})

        
        try:
            response = gemini_model.generate_content(prompt)
            reply = response.text
        except Exception as e:
            reply = f"Error: {e}"

        st.chat_message("assistant").markdown(reply)
        st.session_state["messages"].append({"role": "assistant", "content": reply})

