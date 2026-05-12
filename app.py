
from flask import Flask, render_template, request
import pandas as pd
import joblib
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score

app = Flask(__name__)

# Cargar modelos
logistic_model = joblib.load("logistic_model.pkl")
nn_model = joblib.load("neural_network_model.pkl")
scaler = joblib.load("scaler.pkl")

FEATURES = [
    'mean radius', 'mean texture', 'mean perimeter', 'mean area',
    'mean smoothness', 'mean compactness', 'mean concavity',
    'mean concave points', 'mean symmetry', 'mean fractal dimension',
    'radius error', 'texture error', 'perimeter error', 'area error',
    'smoothness error', 'compactness error', 'concavity error',
    'concave points error', 'symmetry error', 'fractal dimension error',
    'worst radius', 'worst texture', 'worst perimeter', 'worst area',
    'worst smoothness', 'worst compactness', 'worst concavity',
    'worst concave points', 'worst symmetry', 'worst fractal dimension'
]

@app.route("/")
def home():
    return render_template("index.html", features=FEATURES)

@app.route("/predict", methods=["POST"])
def predict():
    model_choice = request.form["model"]

    values = [float(request.form[feature]) for feature in FEATURES]

    df = pd.DataFrame([values], columns=FEATURES)
    scaled = scaler.transform(df)

    model = logistic_model if model_choice == "logistic" else nn_model

    prediction = model.predict(scaled)[0]

    result = "Benigno" if prediction == 1 else "Maligno"

    return render_template("result.html", prediction=result)

@app.route("/batch", methods=["POST"])
def batch():
    file = request.files["file"]
    model_choice = request.form["model"]

    df = pd.read_csv(file)

    X = df.drop("target", axis=1)
    y = df["target"]

    X_scaled = scaler.transform(X)

    model = logistic_model if model_choice == "logistic" else nn_model

    preds = model.predict(X_scaled)

    cm = confusion_matrix(y, preds)
    acc = accuracy_score(y, preds)
    report = classification_report(y, preds)

    return render_template(
        "batch_result.html",
        confusion=cm,
        accuracy=acc,
        report=report
    )

if __name__ == "__main__":
    app.run(debug=True)
