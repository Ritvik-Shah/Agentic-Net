import streamlit as st
import pandas as pd
import os
from ollama import Client
from agentic_tools import create_train_model, predict_model

# Initialize Ollama client
client = Client()

st.title("Agentic Net")

# Ensure datasets folder exists
os.makedirs("datasets", exist_ok=True)

# 1️⃣ Upload dataset
uploaded_file = st.file_uploader("Upload CSV dataset", type="csv")

if uploaded_file:
    # Save uploaded dataset into the datasets folder
    dataset_path = os.path.join("datasets", uploaded_file.name)
    with open(dataset_path, "wb") as f:
        f.write(uploaded_file.getbuffer())

    # Load dataset to preview
    df = pd.read_csv(dataset_path)
    st.write("📊 Dataset Preview", df.head())

    # 2️⃣ Specify network architecture
    layer_input = st.text_input("Enter layer sizes (comma separated, e.g., 32,16,8)", "32,32")
    layers = [int(x.strip()) for x in layer_input.split(",") if x.strip().isdigit()]

    epochs = st.number_input("Epochs", value=10, step=1)
    lr = st.number_input("Learning rate", value=0.001, format="%.5f")

    if st.button("Train Model"):
        # Chat message to LLM (contextual but not critical to training)
        messages = [{"role": "user", "content": f"Train a neural network on {uploaded_file.name} with layers {layers}"}]
        available_functions = {"create_train_model": create_train_model}

        response = client.chat(
            model="gpt-oss:20b-cloud",
            messages=messages,
            tools=list(available_functions.values())
        )

        # Train model directly
        model_id = create_train_model(dataset_path, layers, epochs, lr)
        st.success(f"✅ Model trained successfully! Model ID: {model_id}")

        # Store for predictions later
        st.session_state["last_model_id"] = model_id

# 3️⃣ Predict on new data
st.subheader("Test Predictions")
X_input = st.text_area("Enter rows as comma-separated features (one row per line)")

if st.button("Predict") and "last_model_id" in st.session_state:
    try:
        model_id = st.session_state["last_model_id"]
        rows = [[float(val.strip()) for val in line.split(",")] for line in X_input.strip().split("\n") if line]
        predictions = predict_model(model_id, rows)
        st.write("🧩 Predictions:", predictions)
    except Exception as e:
        st.error(f"Error during prediction: {e}")
