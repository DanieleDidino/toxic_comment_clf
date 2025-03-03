import streamlit as st
import plotly.graph_objects as go
import numpy as np

# Streamlit App Title
st.title("Toxic Comment Classifier")

# Sample Categories
categories = ["Toxic", "Severe Toxic", "Obscene", "Threat", "Insult", "Identity Hate"]
probabilities = np.array([0.49, 0.30, 0.50, 0.60, 0.80, 0.99])
probabilities *= 100 # convert to percentage


with st.expander("What do the categories mean?"):
    st.write("""
    - **Toxic** 🏆 - ADD description here.
    - **Severe Toxic** 🏛️ - ADD description here.
    - **Obscene** 💻 - ADD description here.
    - **Threat** 🏥 - ADD description here.
    - **Insult** 🎬 - ADD description here.
    - **Identity Hate** 💰 - ADD description here.
    """)

# NOT USED ------------------------------------------------------------------------------------
# def values_to_plot(probabilities: np.array, threshold: float = 0.5) -> np.array:
#     if probabilities[0] < threshold:
#         # If toxic is less than threshold, the other categories are "0"
#         new_prob = np.zeros_like(probabilities)
#         new_prob[0] = probabilities[0]
#     else:
#         new_prob = probabilities
#     new_prob *= 100 # convert to percentage
#     return new_prob
# 
# 
# threshold = st.slider("Set a threshold:", min_value=0.0, max_value=1.0, value=0.5, step=0.1)
# st.write(f"Current threshold: {threshold}")  # Display selected value
# 
# probabilities = values_to_plot(probabilities, threshold)
# NOT USED ------------------------------------------------------------------------------------


# Text to evaluate
with st.container():
    text_input = st.text_area("Enter some text for classification:", "This is a sample input.", height=150)

# Arrange in a 3x2 grid
rows, cols = 3, 2 # 2, 3
gauge_width = 0.3  # Fixed width for all gauges
gauge_height = 0.3  # Fixed height for all gauges
spacing_x = (1 - cols * gauge_width) / (cols + 1)  # Horizontal spacing
spacing_y = (1 - rows * gauge_height) / (rows + 1)  # Vertical spacing

# Create figure
fig = go.Figure()
for i, category in enumerate(categories):
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
        title={'text': category},
        gauge={'axis': {'range': [0, 100]}},
        domain={'x': [x_start, x_end], 'y': [y_start, y_end]}
    ))

# Set figure size
fig.update_layout(
    height=800,
    width=1000,
    paper_bgcolor="steelblue"
)

with st.container():
    st.plotly_chart(fig)
