import streamlit as st
import pandas as pd
import time

st.set_page_config(page_title="Crowd Analysis Dashboard", layout="wide")

st.title("🚨 Crowd Behavior Analysis Dashboard")

# Auto refresh every 2 seconds
placeholder = st.empty()

while True:
    with placeholder.container():
        try:
            data = pd.read_csv("report.csv")

            # Latest values
            latest = data.iloc[-1]

            col1, col2, col3 = st.columns(3)

            col1.metric("👥 People Count", int(latest["People"]))
            col2.metric("⚠️ Violence Level", round(latest["Violence"], 2))

            status = "SAFE ✅" if latest["Violence"] < 0.6 else "DANGER ⚠️"
            col3.metric("Status", status)

            st.subheader("📊 Violence Over Time")
            st.line_chart(data["Violence"])

            st.subheader("👥 People Count Over Time")
            st.line_chart(data["People"])

            st.subheader("📄 Recent Data")
            st.dataframe(data.tail(10))

        except:
            st.warning("No data found. Run main program first.")

    time.sleep(2)