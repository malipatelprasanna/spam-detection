from flask import Flask, request, jsonify
import joblib

app = Flask(__name__)

model = joblib.load("models/spam_classifier.pkl")
vectorizer = joblib.load("models/vectorizer.pkl")

@app.route("/")

def home():
    return "Spam Detection API Running"

@app.route("/predict", methods=["POST"])

def predict():

    data = request.json["message"]

    transformed = vectorizer.transform([data])

    prediction = model.predict(transformed)[0]

    result = "Spam" if prediction == 1 else "Ham"

    return jsonify({
        "prediction": result
    })

if __name__ == "__main__":
    app.run(debug=True)