import streamlit as st
import pandas as pd
import pickle
import matplotlib.pyplot as plt
from datetime import datetime


# ---------------------------------------
# Page Configuration
# ---------------------------------------

st.set_page_config(
    page_title="AI Cybersecurity Threat Detector",
    page_icon="🛡️",
    layout="wide"
)


# ---------------------------------------
# Initialize Prediction History
# ---------------------------------------

if "history" not in st.session_state:
    st.session_state.history = []


# ---------------------------------------
# Load Model
# ---------------------------------------

try:

    with open("model.pkl", "rb") as file:
        saved_data = pickle.load(file)

    model = saved_data["model"]
    protocol_encoder = saved_data["protocol_encoder"]
    label_encoder = saved_data["label_encoder"]
    accuracy = saved_data["accuracy"]
    confusion_matrix_data = saved_data["confusion_matrix"]

except FileNotFoundError:

    st.error(
        "model.pkl was not found. "
        "Please run train_model.py first."
    )

    st.stop()


# ---------------------------------------
# Load Dataset
# ---------------------------------------

data = pd.read_csv("dataset.csv")


# ---------------------------------------
# Header
# ---------------------------------------

st.title("🛡️ AI-Based Cybersecurity Threat Detector")

st.write(
    "AI-powered network activity analysis "
    "and potential threat detection."
)

st.divider()


# ---------------------------------------
# Dataset Statistics
# ---------------------------------------

total_records = len(data)

normal_count = len(
    data[data["label"] == "Normal"]
)

threat_count = len(
    data[data["label"] == "Threat"]
)


col1, col2, col3, col4 = st.columns(4)


with col1:
    st.metric(
        "Total Records",
        total_records
    )


with col2:
    st.metric(
        "Normal Records",
        normal_count
    )


with col3:
    st.metric(
        "Threat Records",
        threat_count
    )


with col4:
    st.metric(
        "Model Accuracy",
        f"{accuracy * 100:.2f}%"
    )


st.divider()


# ---------------------------------------
# Sidebar
# ---------------------------------------

st.sidebar.title("🔍 Network Traffic Input")


packet_size = st.sidebar.number_input(
    "Packet Size",
    min_value=1,
    max_value=10000,
    value=500
)


connection_duration = st.sidebar.number_input(
    "Connection Duration",
    min_value=1,
    max_value=1000,
    value=15
)


failed_logins = st.sidebar.number_input(
    "Failed Login Attempts",
    min_value=0,
    max_value=100,
    value=0
)


requests_per_second = st.sidebar.number_input(
    "Requests Per Second",
    min_value=1,
    max_value=1000,
    value=6
)


port_number = st.sidebar.selectbox(
    "Port Number",
    [21, 22, 23, 80, 443]
)


protocol = st.sidebar.selectbox(
    "Protocol",
    ["TCP"]
)


detect_button = st.sidebar.button(
    "🚨 Detect Threat",
    use_container_width=True
)


# ---------------------------------------
# Threat Detection
# ---------------------------------------

if detect_button:

    protocol_encoded = protocol_encoder.transform(
        [protocol]
    )[0]


    input_data = pd.DataFrame(
        [[
            packet_size,
            connection_duration,
            failed_logins,
            requests_per_second,
            port_number,
            protocol_encoded
        ]],
        columns=[
            "packet_size",
            "connection_duration",
            "failed_logins",
            "requests_per_second",
            "port_number",
            "protocol"
        ]
    )


    # Make prediction

    prediction = model.predict(
        input_data
    )


    prediction_label = (
        label_encoder
        .inverse_transform(prediction)[0]
    )


    # Get probabilities

    probabilities = model.predict_proba(
        input_data
    )[0]


    classes = label_encoder.classes_


    probability_dict = dict(
        zip(classes, probabilities)
    )


    threat_probability = (
        probability_dict.get(
            "Threat",
            0
        ) * 100
    )


    # -----------------------------------
    # Determine Risk
    # -----------------------------------

    if threat_probability >= 80:

        risk_level = "HIGH"

    elif threat_probability >= 50:

        risk_level = "MEDIUM"

    else:

        risk_level = "LOW"


    # -----------------------------------
    # Save History
    # -----------------------------------

    history_record = {

        "Time": datetime.now().strftime(
            "%Y-%m-%d %H:%M:%S"
        ),

        "Packet Size": packet_size,

        "Connection Duration":
            connection_duration,

        "Failed Logins":
            failed_logins,

        "Requests / Second":
            requests_per_second,

        "Port":
            port_number,

        "Protocol":
            protocol,

        "Prediction":
            prediction_label,

        "Threat Probability":
            f"{threat_probability:.2f}%",

        "Risk Level":
            risk_level
    }


    st.session_state.history.append(
        history_record
    )


    # -----------------------------------
    # Display Result
    # -----------------------------------

    st.subheader(
        "🔎 Threat Detection Result"
    )


    if prediction_label == "Threat":

        st.error(
            "🚨 POTENTIAL THREAT DETECTED"
        )

    else:

        st.success(
            "✅ NORMAL ACTIVITY"
        )


    # -----------------------------------
    # Probability
    # -----------------------------------

    st.subheader(
        "Threat Probability"
    )


    st.progress(
        min(int(threat_probability), 100)
    )


    st.write(
        f"Threat Probability: "
        f"**{threat_probability:.2f}%**"
    )


    # -----------------------------------
    # Risk Level
    # -----------------------------------

    if risk_level == "HIGH":

        st.error(
            "🔴 Risk Level: HIGH"
        )

        st.warning(
            "Investigate this activity immediately."
        )

    elif risk_level == "MEDIUM":

        st.warning(
            "🟠 Risk Level: MEDIUM"
        )

        st.info(
            "Further investigation is recommended."
        )

    else:

        st.success(
            "🟢 Risk Level: LOW"
        )

        st.info(
            "The activity appears less suspicious "
            "according to this model."
        )


    # -----------------------------------
    # Prediction Details
    # -----------------------------------

    st.subheader(
        "📋 Prediction Details"
    )


    result_table = pd.DataFrame(
        {
            "Parameter": [
                "Packet Size",
                "Connection Duration",
                "Failed Logins",
                "Requests Per Second",
                "Port",
                "Protocol",
                "Prediction",
                "Threat Probability",
                "Risk Level"
            ],

            "Value": [
                packet_size,
                connection_duration,
                failed_logins,
                requests_per_second,
                port_number,
                protocol,
                prediction_label,
                f"{threat_probability:.2f}%",
                risk_level
            ]
        }
    )


    st.table(
        result_table
    )


st.divider()


# ---------------------------------------
# Prediction History
# ---------------------------------------

st.subheader(
    "🕒 Prediction History"
)


if len(st.session_state.history) > 0:

    history_df = pd.DataFrame(
        st.session_state.history
    )


    st.dataframe(
        history_df,
        use_container_width=True
    )


    # -----------------------------------
    # Download History
    # -----------------------------------

    csv_data = history_df.to_csv(
        index=False
    )


    st.download_button(
        label="📥 Download Prediction History",
        data=csv_data,
        file_name="threat_detection_history.csv",
        mime="text/csv"
    )


    # -----------------------------------
    # Clear History
    # -----------------------------------

    if st.button(
        "🗑️ Clear History"
    ):

        st.session_state.history = []

        st.rerun()

else:

    st.info(
        "No predictions yet. "
        "Use the sidebar to detect network activity."
    )


st.divider()


# ---------------------------------------
# Dataset Distribution
# ---------------------------------------

st.subheader(
    "📊 Dataset Distribution"
)


chart_data = pd.DataFrame(
    {
        "Category": [
            "Normal",
            "Threat"
        ],

        "Count": [
            normal_count,
            threat_count
        ]
    }
)


st.bar_chart(
    chart_data.set_index(
        "Category"
    )
)


# ---------------------------------------
# Confusion Matrix
# ---------------------------------------

st.subheader(
    "📈 Confusion Matrix"
)


fig, ax = plt.subplots()


ax.imshow(
    confusion_matrix_data
)


ax.set_xlabel(
    "Predicted"
)


ax.set_ylabel(
    "Actual"
)


ax.set_title(
    "Random Forest Confusion Matrix"
)


for i in range(
    len(confusion_matrix_data)
):

    for j in range(
        len(confusion_matrix_data[i])
    ):

        ax.text(
            j,
            i,
            confusion_matrix_data[i][j],
            ha="center",
            va="center"
        )


st.pyplot(
    fig
)


# ---------------------------------------
# Dataset Preview
# ---------------------------------------

st.subheader(
    "📋 Dataset Preview"
)


st.dataframe(
    data,
    use_container_width=True
)


# ---------------------------------------
# About
# ---------------------------------------

st.divider()


st.subheader(
    "ℹ️ About This Project"
)


st.write(
    """
    This project demonstrates an AI-based cybersecurity
    threat detection system.

    A Random Forest machine-learning model is trained
    using network traffic features including packet size,
    connection duration, failed login attempts,
    requests per second, port number and protocol.

    The trained model predicts whether new network activity
    is Normal or a Potential Threat.

    The application also records prediction history and
    calculates a threat probability and risk level.
    """
)


st.caption(
    "Educational demonstration only — not a production "
    "intrusion detection system."
)