import streamlit as st
import plotly.graph_objects as go
import json
# import numpy as np
import onnxruntime as ort
from utils.predict import predict_onnx, prediction_toxic

####################################################################################
# Variables

CATEGORIES = ["Toxic", "Severe Toxic", "Obscene", "Threat", "Insult", "Identity Hate"]
ONNX_MODEL_PATH = "model/model.onnx"
VOCAB_PATH = "model/vocab_onnx.json"

####################################################################################
# Config streamlit

# About text for the menu item
about = """
Describe here the app
"""

# streamlit config
st.set_page_config(
    page_title="Toxic Comment Classifier",
    layout="wide",
    page_icon=".streamlit/icons8-chatbot-96.png",
    menu_items={
        "About": about
    }
)

st.header("Toxic Comment Classifier")

# Condense the layout
padding = 0
st.markdown(
    f""" <style>
    .reportview-container .main .block-container{{
        padding-top: {padding}rem;
        padding-right: {padding}rem;
        padding-left: {padding}rem;
        padding-bottom: {padding}rem;
    }} </style> """,
    unsafe_allow_html=True,
)

# load custom css styles
with open(".streamlit/custom.css") as f:
    st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)


if "onnx_session" not in st.session_state:
    # Load the ONNX model
    st.session_state.onnx_session = ort.InferenceSession(ONNX_MODEL_PATH)

if "vocab" not in st.session_state:
    # Load vocab
    st.session_state.vocab = json.load(open(VOCAB_PATH))

####################################################################################
# Left column (first part)

sidebar = st.sidebar

with sidebar:

    # Custom page title and subtitle
    #st.title("Toxic Comment Classifier")
    st.subheader("An app for classifying text into toxic categories using a CNN+GRU architecture", divider="orange")
    st.markdown("<br>", unsafe_allow_html=True)
    # st.markdown("<br>", unsafe_allow_html=True)
    # st.markdown("<br>", unsafe_allow_html=True)
    
    # st.markdown("---")  # Add horizontal line
    threshold = st.slider("Set a threshold:", min_value=0.0, max_value=1.0, value=0.5, step=0.1)
    st.write(f"Current threshold: {threshold}")

    # Category description
    # st.markdown("---")  # Add horizontal line
    st.markdown("<hr style='border: 2px solid #FFBF00;'>", unsafe_allow_html=True) # Add horizontal line
    with st.expander("What do the categories mean?"):
        st.write(
            """
            - **Toxic**: General toxic or rude comments that are likely to be offensive.
            - **Severe Toxic**: Extremely aggressive, hateful, or threatening comments.
            - **Obscene**: Contains offensive language, profanity, or vulgar words.
            - **Threat**: Direct threats of violence or harm towards individuals or groups.
            - **Insult**: Personal attacks, name-calling, or derogatory remarks.
            - **Identity Hate**: Hate speech targeting specific identity groups (race, gender, religion, etc.).
            """
        )
    
    #st.markdown("---")  # Add horizontal line
    st.markdown("<hr style='border: 2px solid #FFBF00;'>", unsafe_allow_html=True)
    st.markdown("### 📞 Contact & Links")
    st.write("**📧 Email:** daniele.didino@gmail.com")  
    st.write("**🔗 [LinkedIn](https://www.linkedin.com/feed/)**")
    st.write("**💻 [GitHub](https://github.com/DanieleDidino)**")
    st.write("**🌐 [website](https://danieledidino.github.io/)**")

####################################################################################
# Right side: text to classify & gauges

# Text to evaluate
with st.container():
    text_input = st.text_area("Enter some text for classification:", "This is a sample input.", height=150)

probabilities = predict_onnx(
    text_input,
    vocab=st.session_state.vocab,
    session=st.session_state.onnx_session)
probabilities *= 100 # convert to percentage

pred = prediction_toxic(probabilities[0], threshold)
st.info(f"The text has been classified as **{pred}** \n\n Probability: **{probabilities[0]:.1f}%**")

# Figure description
st.markdown("### Classification Probabilities 📊")

# Arrange in a 3x2 grid
rows, cols = 2, 3 # 3, 2
gauge_width = 0.3  # Fixed width for all gauges
gauge_height = 0.3  # Fixed height for all gauges
spacing_x = (1 - cols * gauge_width) / (cols + 1)  # Horizontal spacing
spacing_y = (1 - rows * gauge_height) / (rows + 1)  # Vertical spacing

# Create figure
fig = go.Figure()
for i, category in enumerate(CATEGORIES):
    row = i // cols  # 0 or 1
    col = i % cols   # 0, 1, 2

    # Compute domain
    x_start = spacing_x + col * (gauge_width + spacing_x)
    x_end = x_start + gauge_width
    y_start = 1 - (row + 1) * (gauge_height + spacing_y)
    y_end = y_start + gauge_height

    # Create Gauge plot
    fig.add_trace(go.Indicator(
        mode="gauge+number",
        value=probabilities[i],
        title={"text": category},
        gauge={
            "axis": {"range": [0, 100]},
            "bar": {"color": "red" if probabilities[i] > 50 else "green"},
        },
        number={"font": {"color": "red" if probabilities[i] > 50 else "green"}},
        domain={"x": [x_start, x_end], "y": [y_start, y_end]}
    ))

# Set figure size
fig.update_layout(
    height=800,
    width=1000,
    #paper_bgcolor="steelblue"
)

with st.container():
    st.plotly_chart(fig)

st.write(
    "The gauge charts below display the probability of the entered text belonging to each category. "
    "The values range from 0% to 100%, where higher values indicate stronger classification confidence."
)
