from sklearn.datasets import (
    make_classification,
    make_circles,
    make_blobs,
    make_moons
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import streamlit as st

@st.cache_data
def load_dataset(name):
    if name == "Classification":
        return make_classification(
            n_samples=300,
            n_features=2,
            n_informative=2,
            n_redundant=0,
            random_state=42
        )
    if name == "Circles":
        return make_circles(n_samples=300, noise=0.1, factor=0.5, random_state=42)
    if name == "Blobs":
        return make_blobs(n_samples=300, centers=3, random_state=42)
    if name == "Moons":
        return make_moons(n_samples=300, noise=0.1, random_state=42)

@st.cache_data
def split_and_scale(X, y):
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    return train_test_split(
        X_scaled,
        y,
        test_size=0.2,
        random_state=42
    )
