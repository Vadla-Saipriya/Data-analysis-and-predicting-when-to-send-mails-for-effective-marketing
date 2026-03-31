# ================== 🔥 PREMIUM STREAMLIT APP 🔥 ================== #

import streamlit as st
import pickle
import numpy as np
import pandas as pd
import plotly.express as px

# ================== PAGE CONFIG ================== #
st.set_page_config(page_title="Email Dashboard", layout="wide")

# ================== CUSTOM CSS (LIGHT ORANGE THEME) ================== #
st.markdown("""
<style>

/* Full Background */
.stApp {
    background: linear-gradient(to right, #ffe0cc, #ffcc99);
}

/* Sidebar */
section[data-testid="stSidebar"] {
    background: #ffb380;
    color: black;
}

/* Cards */
.card {
    background: #fff5e6;
    padding: 20px;
    border-radius: 15px;
    box-shadow: 0px 4px 15px rgba(0,0,0,0.1);
}

/* Titles */
h1, h2, h3 {
    color: #5a2d0c;
}

/* Buttons */
.stButton>button {
    background-color: #ff7f50;
    color: white;
    border-radius: 10px;
    height: 45px;
    width: 100%;
}

/* Metrics */
[data-testid="stMetric"] {
    background: #fff5e6;
    padding: 15px;
    border-radius: 12px;
    box-shadow: 0px 2px 10px rgba(0,0,0,0.08);
}

/* Input boxes */
.stSelectbox, .stSlider, .stNumberInput {
    background-color: #fff5e6;
}

</style>
""", unsafe_allow_html=True)


# ================== LOAD FILES ================== #
with open("xgboost_model.pkl", "rb") as f:
    model = pickle.load(f)

with open("scaler.pkl", "rb") as f:
    scaler = pickle.load(f)

with open("encoders.pkl", "rb") as f:
    encoders = pickle.load(f)

# ================== SIDEBAR ================== #
st.sidebar.title("📊 Dashboard")
page = st.sidebar.radio("", ["🏠 Home", "📈 Analytics"])

# ================== HOME PAGE ================== #
if page == "🏠 Home":

    st.title("📧 Email Engagement Dashboard")
    st.write("Predict customer engagement in a smart and interactive way.")

    col1, col2 = st.columns([2,1])

    with col1:
        st.markdown("### 📥 Enter Email Details")
        
        st.markdown("<h4 style='color:#bf360c;'>📧 Email Text</h4>", unsafe_allow_html=True)
        email_text = st.selectbox("", encoders['email_text'].classes_)

        st.markdown("<h4 style='color:#bf360c;'>📧 Email Version</h4>", unsafe_allow_html=True)
        email_version = st.selectbox("", encoders['email_version'].classes_)

        st.markdown("<h4 style='color:#bf360c;'>📅 Weekday</h4>", unsafe_allow_html=True)
        weekday = st.selectbox("", encoders['weekday'].classes_)

        st.markdown("<h4 style='color:#bf360c;'>🌍 User Country</h4>", unsafe_allow_html=True)
        user_country = st.selectbox("", encoders['user_country'].classes_)

        st.markdown("<h4 style='color:#bf360c;'>⏰ Hour</h4>", unsafe_allow_html=True)
        hour = st.slider("", 0, 23, 10)

        st.markdown("<h4 style='color:#bf360c;'>🛒 Past Purchases</h4>", unsafe_allow_html=True)
        user_past_purchases = st.number_input("", 0, 100, 1)

        if st.button("🚀 Predict Engagement"):

            email_text_enc = encoders['email_text'].transform([email_text])[0]
            email_version_enc = encoders['email_version'].transform([email_version])[0]
            weekday_enc = encoders['weekday'].transform([weekday])[0]
            user_country_enc = encoders['user_country'].transform([user_country])[0]

            input_data = np.array([[email_text_enc, email_version_enc, weekday_enc, user_country_enc, hour, user_past_purchases]])

            input_data = scaler.transform(input_data)
            prediction = model.predict(input_data)

            with col2:
                st.markdown("### 📊 Prediction Result")

                if prediction[0] == 1:
                    st.success("✨ High Engagement Expected")
                else:
                    st.error("⚠️ Low Engagement Expected")

# ================== ANALYTICS PAGE ================== #
elif page == "📈 Analytics":

    st.title("📊 Analytics Dashboard")

    df = pd.read_csv("cleaneddata.csv")

    df['engagement_status'] = df['engagement_status'].replace({
        'Clicked and Opened': 1,
        'Opened but Not Clicked': 1,
        'Not Opened': 0
    })

    # ================== KPI ================== #
    col1, col2, col3 = st.columns(3)

    conversion_rate = round(df['engagement_status'].mean() * 100, 2)
    total_users = len(df)
    avg_purchases = round(df['user_past_purchases'].mean(), 2)

    col1.metric("📈 Engagement Rate", f"{conversion_rate}%")
    col2.metric("📧 Total Emails", total_users)
    col3.metric("🛒 Avg Purchases", avg_purchases)

    st.markdown("---")

    col1, col2 = st.columns(2)

    # ===== LINE CHART ===== #
    with col1:
        st.markdown("### 📉 Engagement Trend (Hourly)")

        hourly = df.groupby("hour")["engagement_status"].mean().reset_index()
        hourly["engagement_status"] *= 100

        fig1 = px.line(hourly, x="hour", y="engagement_status", markers=True)
        fig1.update_layout(template="plotly_white")

        st.plotly_chart(fig1, use_container_width=True)

    # ===== BAR CHART ===== #
    with col2:
        st.markdown("### 📊 Engagement by Weekday")

        weekday_data = df.groupby("weekday")["engagement_status"].mean().reset_index()
        weekday_data["engagement_status"] *= 100

        fig2 = px.bar(weekday_data, x="weekday", y="engagement_status")
        fig2.update_layout(template="plotly_white")

        st.plotly_chart(fig2, use_container_width=True)

    # ===== COUNTRY ===== #
    st.markdown("### 🌍 Engagement by Country")

    country_data = df.groupby("user_country")["engagement_status"].mean().reset_index()
    country_data["engagement_status"] *= 100

    fig3 = px.bar(country_data, x="user_country", y="engagement_status")
    fig3.update_layout(template="plotly_white")

    st.plotly_chart(fig3, use_container_width=True)