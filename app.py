import streamlit as st
import joblib
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris

model = joblib.load("iris_model.pkl")

st.title("Iris Flower Prediction")

sl = st.number_input("Sepal Length", value=5.1)
sw = st.number_input("Sepal Width", value=3.5)
pl = st.number_input("Petal Length", value=1.4)
pw = st.number_input("Petal Width", value=0.2)

if st.button("Predict"):
    pred = model.predict([[sl, sw, pl, pw]])
    flowers = ["Setosa", "Versicolor", "Virginica"]
    st.success(f"Predicted Flower: {flowers[pred[0]]}")
iris = load_iris()
X = iris.data
y = iris.target

# Create scatter plot
fig, ax = plt.subplots(figsize=(10, 6))

# Plot dataset samples
ax.scatter(
    X[:, 0],
    X[:, 2],
    color="blue",
    label="Actual Data",
    s=40
)

# Highlight predicted sample
ax.scatter(
    sepal_length,
    petal_length,
    color="red",
    s=200,
    label="Prediction"
)

# Labels
ax.set_xlabel("Sepal Length (cm)")
ax.set_ylabel("Petal Length (cm)")
ax.set_title("Iris Flower Prediction")

# Legend and grid
ax.legend()
ax.grid(axis="y", alpha=0.3)

# Display graph in Streamlit
st.pyplot(fig)