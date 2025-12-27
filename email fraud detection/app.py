from flask import Flask, render_template, request
import pickle

# Load trained model and vectorizer
model = pickle.load(open("email_model.pkl", "rb"))
vectorizer = pickle.load(open("vectorizer.pkl", "rb"))

app = Flask(__name__)

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    email_text = request.form["email"]

    text_vector = vectorizer.transform([email_text])
    prediction = model.predict(text_vector)

    result = "🚨 Fraud / Spam Email" if prediction[0] == 1 else "✅ Legitimate Email"

    return render_template("index.html", prediction=result, email=email_text)

if __name__ == "__main__":
    app.run(debug=True)
