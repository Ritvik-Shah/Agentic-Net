import streamlit as st
import pandas as pd
import os
import json
from ollama import Client
from agentic_tools import create_train_model, predict_model

# Initialize Ollama client
client = Client()

st.title("Agentic Net – Train, Predict & Analyze")

# Ensure datasets folder exists
os.makedirs("datasets", exist_ok=True)

# Upload dataset
uploaded_file = st.file_uploader("Upload CSV dataset", type="csv")

if uploaded_file:
    dataset_path = os.path.join("datasets", uploaded_file.name)
    with open(dataset_path, "wb") as f:
        f.write(uploaded_file.getbuffer())

    # Load dataset
    df = pd.read_csv(dataset_path)
    st.write("Dataset Preview", df.head())

    # Save dataset summary for insight context
    dataset_summary = df.describe(include="all").to_dict()

    # Network parameters
    layer_input = st.text_input("Enter layer sizes (comma separated, e.g., 32,16,8)", "32,32")
    layers = [int(x.strip()) for x in layer_input.split(",") if x.strip().isdigit()]

    epochs = st.number_input("Epochs", value=10, step=1)
    lr = st.number_input("Learning rate", value=0.001, format="%.5f")

    if st.button("Train Model"):
        messages = [{"role": "user", "content": f"Train a neural network on {uploaded_file.name} with layers {layers}"}]
        available_functions = {"create_train_model": create_train_model}

        # Call LLM to provide commentary
        try:
            response = client.chat(
                model="gpt-oss:20b-cloud",
                messages=messages,
                tools=list(available_functions.values())
            )
        except Exception as e:
            st.warning(f"Skipping LLM commentary: {e}")

        model_id = create_train_model(dataset_path, layers, epochs, lr)
        st.success(f"Model trained and saved with ID: {model_id}")

        st.session_state["last_model_id"] = model_id
        st.session_state["dataset_summary"] = dataset_summary

# Prediction interface
st.subheader("Make Predictions")
X_input = st.text_area("Enter rows as comma-separated features (one row per line)")

if st.button("Predict") and "last_model_id" in st.session_state:
    try:
        model_id = st.session_state["last_model_id"]
        rows = [[float(val.strip()) for val in line.split(",")] for line in X_input.strip().split("\n") if line]
        predictions = predict_model(model_id, rows)
        st.write("Predictions:", predictions)
        st.session_state["last_predictions"] = predictions
    except Exception as e:
        st.error(f"Error during prediction: {e}")

# Insight query
st.subheader("Ask for Insights")
insight_question = st.text_area(
    "Ask the AI about your dataset or model (e.g., 'Which features seem most influential?' or 'Explain what the model learned.')"
)

if st.button("Get Insight"):
    if not insight_question.strip():
        st.warning("Please enter a question for the AI.")
    else:
        # Gather available context
        context = {
            "dataset_summary": st.session_state.get("dataset_summary", {}),
            "predictions": st.session_state.get("last_predictions", []),
            "architecture": st.session_state.get("last_layers", []),
        }

        # Prepare messages for LLM
        messages = [
            {"role": "system", "content": "You are an expert ML assistant that explains model behavior and dataset patterns clearly."},
            {"role": "user", "content": f"The user asked: {insight_question}"},
            {"role": "user", "content": f"Here is the dataset summary and context:\n{json.dumps(context, indent=2)}"}
        ]

        try:
            response = client.chat(model="gpt-oss:20b-cloud", messages=messages)
            st.markdown("Insight")
            st.write(response["message"]["content"])
        except Exception as e:
            st.error(f"Error generating insight: {e}")
