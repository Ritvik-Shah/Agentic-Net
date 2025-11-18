import os
import json
import pandas as pd
import numpy as np
import streamlit as st
import matplotlib.pyplot as plt
from ollama import Client
from agentic_tools import create_train_model, predict_model, load_model_metadata

# Initialize
client = Client()
st.set_page_config(page_title="Agentic Net", layout="wide")

st.title("Agentic Net")

# ensure folders
os.makedirs("datasets", exist_ok=True)
os.makedirs("models", exist_ok=True)

# Sidebar: global options
st.sidebar.header("Global Options")
scaler_type = st.sidebar.selectbox("Scaler", ["standard", "minmax", "none"], index=0)
test_size = st.sidebar.slider("Validation fraction", 0.05, 0.5, 0.2, 0.05)
auto_search = st.sidebar.checkbox("Use quick AutoML search (short)", value=False)

# Upload dataset
uploaded_file = st.file_uploader("Upload CSV dataset (last column is label)", type=["csv"])
if uploaded_file:
    dataset_path = os.path.join("datasets", uploaded_file.name)
    with open(dataset_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
    df = pd.read_csv(dataset_path)
    st.write("Dataset preview:")
    st.dataframe(df.head())

    # network UI
    st.subheader("Network & Training")
    col1, col2 = st.columns([2, 1])
    with col1:
        layer_input = st.text_input("Layers (comma separated)", "64,32")
        layers = [int(x.strip()) for x in layer_input.split(",") if x.strip().isdigit()]
        activation = st.selectbox("Activation", ["relu", "tanh"], index=0)
        dropout = st.number_input("Dropout (0-1)", min_value=0.0, max_value=0.9, value=0.0, step=0.05)
    with col2:
        epochs = st.number_input("Epochs", value=10, step=1)
        lr = st.number_input("Learning rate", value=0.001, format="%.6f")
        batch_size = st.number_input("Batch size", value=32, step=1)

    if st.button("Train"):
        try:
            meta = create_train_model(
                dataset_path=dataset_path,
                layers=layers,
                epochs=epochs,
                lr=lr,
                batch_size=batch_size,
                test_size=test_size,
                scaler_type=scaler_type if scaler_type != "none" else None,
                activation=activation,
                dropout=dropout,
                auto_search=auto_search
            )
            st.success(f"Model trained and saved. Model ID: {meta['model_id']}")
            st.session_state["last_model_id"] = meta["model_id"]
            st.session_state["dataset_head"] = df.head(10).to_dict()
            st.session_state["last_metadata"] = meta
        except Exception as e:
            st.error(f"Training failed: {e}")

# If we have a trained model in session, show metadata + metrics
if "last_model_id" in st.session_state:
    model_id = st.session_state["last_model_id"]
    st.subheader("Model Metadata and Evaluation")
    try:
        meta = load_model_metadata(model_id)
        st.json(meta, expanded=False)
        # show val metrics summary
        val_metrics = meta.get("val_metrics", {})
        if val_metrics:
            st.write("Validation accuracy:", val_metrics.get("accuracy"))
            cm = val_metrics.get("confusion_matrix")
            if cm:
                fig, ax = plt.subplots()
                ax.imshow(np.array(cm))
                ax.set_title("Confusion Matrix")
                st.pyplot(fig)
        # show permutation importances
        if meta.get("permutation_importance") is not None:
            imp = meta["permutation_importance"]
            st.write("Permutation importance (validation):")
            st.bar_chart(imp["mean"])
    except Exception as e:
        st.warning(f"Could not load metadata: {e}")

# Prediction panel
st.subheader("Predict")
rows_text = st.text_area("Enter rows as comma-separated features (one row per line). No header.")
if st.button("Run Predictions"):
    if "last_model_id" not in st.session_state:
        st.warning("Train a model first.")
    else:
        try:
            rows = [list(map(float, r.split(","))) for r in rows_text.strip().split("\n") if r.strip()]
            res = predict_model(st.session_state["last_model_id"], rows)
            st.session_state["last_predictions"] = res
            st.write("Predictions:", res["predictions"])
            st.write("Probabilities (first row):", res["probabilities"][0] if res["probabilities"] else None)
            # quick chart of predicted class counts
            counts = pd.Series(res["predictions"]).value_counts().sort_index()
            st.bar_chart(counts)
        except Exception as e:
            st.error(f"Prediction error: {e}")

# Persisted view of last predictions
if "last_predictions" in st.session_state:
    st.write("Last predictions (persisted):")
    st.write(st.session_state["last_predictions"]["predictions"])

# Insight / chat with model
st.subheader("Ask the Agent about data or model")
question = st.text_area("Ask a question (e.g., which features matter most, are predictions reliable, what patterns exist?)")
if st.button("Ask Agent"):
    if not question.strip():
        st.warning("Please enter a question.")
    else:
        # build context
        context = {}
        if "last_metadata" in st.session_state:
            context["metadata"] = st.session_state["last_metadata"]
        elif "last_model_id" in st.session_state:
            try:
                context["metadata"] = load_model_metadata(st.session_state["last_model_id"])
            except Exception:
                context["metadata"] = {}
        context["predictions"] = st.session_state.get("last_predictions", {})
        context["dataset_sample"] = st.session_state.get("dataset_head", {})

        prompt = f"""
You are an expert ML analyst. The user asks: {question}
Context:
Metadata: {json.dumps(context.get('metadata', {}), indent=2)}
Dataset sample: {json.dumps(context.get('dataset_sample', {}), indent=2)}
Recent predictions: {json.dumps(context.get('predictions', {}), indent=2)}
Please answer concisely and reference the context where appropriate. If you need more info, say exactly what you need.
"""
        try:
            response = client.chat(model="gpt-oss:20b-cloud", messages=[{"role": "user", "content": prompt}])
            # access returned text in a defensive way
            content = ""
            if isinstance(response, dict):
                content = response.get("message", {}).get("content", "")
            else:
                # Ollama client may return Response object with .message
                try:
                    content = response.message["content"]
                except Exception:
                    content = str(response)
            st.markdown("Agent response:")
            st.write(content)
        except Exception as e:
            st.error(f"LLM call failed: {e}")

st.markdown("---")
st.write("Model files are saved under the `models/` directory. Metadata and scalers are stored alongside model weights.")
