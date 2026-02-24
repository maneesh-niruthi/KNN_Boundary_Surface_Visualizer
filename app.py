import streamlit as st
from src.data import load_dataset, split_and_scale
from src.model import train_knn, evaluate_model
from src.plot import plot_decision_boundary

st.set_page_config(page_title="KNN Visualizer", layout="wide")

st.image("assets/knn.png", width=180)
st.title("KNN Decision Boundary Visualizer")

# Sidebar
st.sidebar.header("Model Settings")
k = st.sidebar.slider("Neighbors (k)", 1, 15, 5)
weights = st.sidebar.selectbox("Weights", ["uniform", "distance"])
metric = st.sidebar.selectbox("Metric", ["euclidean", "manhattan", "minkowski"])

st.sidebar.header("Dataset")
dataset = st.sidebar.selectbox(
    "Choose Dataset",
    ["Classification", "Circles", "Blobs", "Moons"]
)

show_test = st.sidebar.checkbox("Show Test Data Only", False)

# Data
X, y = load_dataset(dataset)
X_train, X_test, y_train, y_test = split_and_scale(X, y)

# Model
model = train_knn(X_train, y_train, k, weights, metric)

# Evaluation
acc, cm = evaluate_model(model, X_test, y_test)

col1, col2 = st.columns([2, 1])


with col1:
    plot_decision_boundary(
        model,
        X_test if show_test else X,
        y_test if show_test else y,
        k,
        weights
    )

with col2:
    st.subheader("Model Performance")
    st.metric("Accuracy", f"{acc * 100:.2f}%")
    st.write("Confusion Matrix")
    st.dataframe(cm)
