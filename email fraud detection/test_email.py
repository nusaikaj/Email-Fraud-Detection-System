import pickle

# Load saved files
model = pickle.load(open("email_model.pkl", "rb"))
vectorizer = pickle.load(open("vectorizer.pkl", "rb"))

def predict_email(text):
    text_vector = vectorizer.transform([text])
    prediction = model.predict(text_vector)
    return "Fraud / Spam" if prediction[0] == 1 else "Legitimate"

# Test emails
print(predict_email("Congratulations! You won a free lottery prize"))
print(predict_email("Hi, please find the meeting agenda attached"))
