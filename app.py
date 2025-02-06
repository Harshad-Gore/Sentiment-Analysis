from flask import Flask, request, render_template
import pickle

app = Flask(__name__)

# Load trained model & vectorizer
with open("sentiment_model.pkl", "rb") as f:
    model = pickle.load(f)
with open("tfidf_vectorizer.pkl", "rb") as f:
    vectorizer = pickle.load(f)

@app.route('/', methods=['GET', 'POST'])
def home():
    result = None  # Default value

    if request.method == 'POST':
        text = request.form['text']
        
        # Transform input using TF-IDF
        text_vectorized = vectorizer.transform([text])

        # Predict sentiment and probability
        prediction = model.predict(text_vectorized)[0]
        probabilities = model.predict_proba(text_vectorized)

        # Extract confidence score for the predicted class
        confidence = max(probabilities[0])  # Get the highest probability score

        # Map sentiment values
        sentiment_map = {1: "Positive 😀", 0: "Neutral 😐", -1: "Negative 😞"}
        sentiment_label = sentiment_map.get(prediction, "Unknown")

        # Store the result
        result = {"label": sentiment_label, "score": f"{confidence:.2f}"}  # Round confidence

    return render_template('index.html', result=result, text=text if request.method == 'POST' else '')


if __name__ == '__main__':
    app.run(debug=True)
