import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import streamlit as st

def plot_decision_boundary(model, X, y, k, weights):
    h = 0.02
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1

    xx, yy = np.meshgrid(
        np.arange(x_min, x_max, h),
        np.arange(y_min, y_max, h)
    )


    Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)

    fig, ax = plt.subplots(figsize=(7, 5))
    cmap_light = ListedColormap(["#FFCCCC", "#CCCCFF", "#CCFFCC"])
    cmap_bold = ["red", "blue", "green"]

    ax.contourf(xx, yy, Z, cmap=cmap_light, alpha=0.4)
    ax.scatter(X[:, 0], X[:, 1], c=y, cmap=ListedColormap(cmap_bold), edgecolor="k", s=30)

    ax.set_title(f"KNN Boundary (k={k}, weights={weights})")
    ax.set_xlabel("Feature 1")
    ax.set_ylabel("Feature 2")

    st.pyplot(fig)
    plt.close(fig)
