from flask import Flask, request, render_template
from transformers import pipeline

app = Flask(__name__)

# Load the pre-trained sentiment analysis model
model = pipeline('sentiment-analysis')

@app.route('/', methods=['GET', 'POST'])
def home():
    if request.method == 'POST':
        # Get user input from the form
        text = request.form['text']
        
        # Perform sentiment analysis
        result = model(text)[0]  # The model returns a list of results, so we take the first one

        # Render the result on the same page
        return render_template('index.html', result=result, text=text)
    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)