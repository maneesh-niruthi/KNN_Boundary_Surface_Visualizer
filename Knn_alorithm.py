import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_classification, make_circles, make_blobs, make_moons
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from matplotlib.colors import ListedColormap

# LOGO
st.image("knn.png", width=200)

# Streamlit App Title
st.title("KNN Boundary Surface Visualization")

# Sidebar for Parameters
st.sidebar.header("KNN Parameters")
k_neighbors = st.sidebar.slider("Number of Neighbors (k)", 1, 15, 5)
weights = st.sidebar.selectbox("Weights", ["uniform", "distance"], index=0)
metric = st.sidebar.selectbox("Distance Metric", ["euclidean", "manhattan", "minkowski"], index=2)

# Sidebar for Dataset Selection
dataset_name = st.sidebar.selectbox("Select Dataset", ["Classification", "Circles", "Blobs", "Moons"])

# Generate Dataset
def load_dataset(name):
    if name == "Classification":
        return make_classification(n_samples=300, n_features=2, n_informative=2, n_redundant=0, random_state=42)
    elif name == "Circles":
        return make_circles(n_samples=300, noise=0.1, factor=0.5, random_state=42)
    elif name == "Blobs":
        return make_blobs(n_samples=300, centers=3, random_state=42)
    elif name == "Moons":
        return make_moons(n_samples=300, noise=0.1, random_state=42)
    else:
        return None

X, y = load_dataset(dataset_name)
X = StandardScaler().fit_transform(X)  # Standardizing features
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train KNN Model
knn = KNeighborsClassifier(n_neighbors=k_neighbors, weights=weights, metric=metric)
knn.fit(X_train, y_train)

# Plot Decision Boundary
def plot_decision_boundary(model, X, y):
    h = 0.02  # Mesh step size
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
    Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    cmap_light = ListedColormap(["#FFAAAA", "#AAAAFF", "#AAFFAA"])
    cmap_bold = ["r", "b", "g"]
    
    
    plt.figure(figsize=(8, 6))
    plt.contourf(xx, yy, Z, alpha=0.3, cmap=cmap_light)
    plt.scatter(X[:, 0], X[:, 1], c=y, edgecolors="k", cmap=ListedColormap(cmap_bold))
    plt.xlabel("Feature 1")
    plt.ylabel("Feature 2")
    plt.title(f"KNN Decision Boundary (k={k_neighbors}, weights={weights})")
    st.pyplot(plt)

plot_decision_boundary(knn, X, y)