# app.py

import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import load_iris

# Train and save model if not exists
def train_and_save_model():
    iris = load_iris()
    X = iris.data
    y = iris.target
    model = RandomForestClassifier(random_state=42)
    model.fit(X, y)
    with open("iris_model.pkl", "wb") as f:
        pickle.dump(model, f)

# Load saved model
def load_model():
    try:
        with open("iris_model.pkl", "rb") as f:
            model = pickle.load(f)
    except FileNotFoundError:
        train_and_save_model()
        with open("iris_model.pkl", "rb") as f:
            model = pickle.load(f)
    return model

# Main Streamlit app
def main():
    st.set_page_config(page_title="Iris Classifier", layout="centered")
    st.title("🌸 Iris Flower Classifier")
    st.markdown("This app predicts the species of an Iris flower based on its features.")

    # Sidebar input
    st.sidebar.header("Input Features")
    sepal_length = st.sidebar.slider("Sepal Length (cm)", 4.0, 8.0, 5.1)
    sepal_width = st.sidebar.slider("Sepal Width (cm)", 2.0, 4.5, 3.5)
    petal_length = st.sidebar.slider("Petal Length (cm)", 1.0, 7.0, 1.4)
    petal_width = st.sidebar.slider("Petal Width (cm)", 0.1, 2.5, 0.2)

    input_data = np.array([[sepal_length, sepal_width, petal_length, petal_width]])

    # Load model and predict
    model = load_model()
    prediction = model.predict(input_data)
    prediction_proba = model.predict_proba(input_data)

    # Display prediction
    iris = load_iris()
    st.subheader("Prediction")
    st.write(f"🌼 Predicted species: **{iris.target_names[prediction[0]]}**")

    # Display probabilities
    st.subheader("Prediction Probability")
    proba_df = pd.DataFrame(prediction_proba, columns=iris.target_names)
    st.bar_chart(proba_df.T)

    # Scatter plot of features
    st.subheader("Feature Visualization")
    iris_df = pd.DataFrame(iris.data, columns=iris.feature_names)
    iris_df['species'] = pd.Categorical.from_codes(iris.target, iris.target_names)

    fig, ax = plt.subplots()
    sns.scatterplot(
        data=iris_df,
        x="petal length (cm)",
        y="petal width (cm)",
        hue="species",
        palette="Set2",
        ax=ax
    )
    ax.scatter(petal_length, petal_width, color="black", label="Your Input", marker="X", s=100)
    ax.legend()
    st.pyplot(fig)

if __name__ == "__main__":
    main()
