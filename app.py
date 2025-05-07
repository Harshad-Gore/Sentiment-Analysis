from flask import Flask, request, render_template, redirect, url_for, send_file, session
import pickle
import pandas as pd
import os
import io
from fpdf import FPDF
import matplotlib.pyplot as plt
import tempfile

app = Flask(__name__)
app.secret_key = 'your_secret_key_here'  # Needed for session

# Load trained model & vectorizer
with open("sentiment_model.pkl", "rb") as f:
    model = pickle.load(f)
with open("tfidf_vectorizer.pkl", "rb") as f:
    vectorizer = pickle.load(f)

@app.route('/', methods=['GET', 'POST'])
def home():
    result = session.get('result')
    text = session.get('text', '')
    text_error = None
    text_submitted = False
    if request.method == 'POST':
        text_submitted = True
        if 'text' in request.form:
            text = request.form['text']
            if not text.strip():
                text_error = 'Please enter some text for analysis.'
                session['result'] = None
                session['text'] = ''
            else:
                text_vectorized = vectorizer.transform([text])
                prediction = model.predict(text_vectorized)[0]
                probabilities = model.predict_proba(text_vectorized)
                confidence = max(probabilities[0])
                sentiment_map = {1: "Positive 😀", 0: "Neutral 😐", -1: "Negative 😞"}
                sentiment_label = sentiment_map.get(prediction, "Unknown")
                result = {"label": sentiment_label, "score": round(confidence, 5)}
                session['result'] = result
                session['text'] = text
    batch_results = session.get('batch_results')
    chart_data = session.get('chart_data')
    analytics = session.get('analytics')
    batch_error = None
    batch_submitted = False
    return render_template('index.html', result=result, text=text, batch_results=batch_results, chart_data=chart_data, text_error=text_error, batch_error=batch_error, analytics=analytics, text_submitted=text_submitted, batch_submitted=batch_submitted)

@app.route('/batch', methods=['GET', 'POST'])
def batch():
    batch_results = None
    chart_data = None
    batch_error = None
    analytics = None
    batch_submitted = False
    if request.method == 'POST':
        batch_submitted = True
        if 'csv_file' not in request.files:
            batch_error = 'No file part. Please upload a CSV file.'
        else:
            file = request.files['csv_file']
            if file.filename == '':
                batch_error = 'No file selected. Please choose a CSV file to upload.'
            elif not file.filename.lower().endswith('.csv'):
                batch_error = 'Invalid file type. Please upload a valid CSV file.'
            else:
                try:
                    df = pd.read_csv(file)
                    text_col = df.columns[0]
                    texts = df[text_col].astype(str).tolist()
                    text_vectors = vectorizer.transform(texts)
                    predictions = model.predict(text_vectors)
                    probabilities = model.predict_proba(text_vectors)
                    sentiment_map = {1: "Positive 😀", 0: "Neutral 😐", -1: "Negative 😞"}
                    batch_results = []
                    sentiment_counts = {"Positive 😀": 0, "Neutral 😐": 0, "Negative 😞": 0}
                    confidences = []
                    for i, text in enumerate(texts):
                        pred = predictions[i]
                        prob = max(probabilities[i])
                        label = sentiment_map.get(pred, "Unknown")
                        batch_results.append({
                            'text': text,
                            'label': label,
                            'score': round(prob, 5)
                        })
                        confidences.append(prob)
                        if label in sentiment_counts:
                            sentiment_counts[label] += 1
                    chart_data = sentiment_counts
                    total = sum(sentiment_counts.values())
                    analytics = {
                        'total': total,
                        'positive_pct': round(sentiment_counts["Positive 😀"] / total * 100, 2) if total else 0,
                        'neutral_pct': round(sentiment_counts["Neutral 😐"] / total * 100, 2) if total else 0,
                        'negative_pct': round(sentiment_counts["Negative 😞"] / total * 100, 2) if total else 0,
                        'avg_conf': round(sum(confidences) / len(confidences), 4) if confidences else 0
                    }
                    # Save batch results to CSV in memory for download
                    df_report = pd.DataFrame(batch_results)
                    csv_buffer = io.StringIO()
                    df_report.to_csv(csv_buffer, index=False)
                    csv_buffer.seek(0)
                    global last_csv_report
                    last_csv_report = csv_buffer.getvalue()
                    # Store in session
                    session['batch_results'] = batch_results
                    session['chart_data'] = chart_data
                    session['analytics'] = analytics
                except Exception as e:
                    batch_error = f"Error processing file: {str(e)}"
                    session['batch_results'] = None
                    session['chart_data'] = None
                    session['analytics'] = None
    result = session.get('result')
    text = session.get('text', '')
    text_error = None
    text_submitted = False
    return render_template('index.html', result=result, text=text, batch_results=batch_results, chart_data=chart_data, text_error=text_error, batch_error=batch_error, analytics=analytics, text_submitted=text_submitted, batch_submitted=batch_submitted)

@app.route('/download_report')
def download_report():
    global last_csv_report
    if last_csv_report:
        return send_file(io.BytesIO(last_csv_report.encode()), mimetype='text/csv', as_attachment=True, download_name='sentiment_report.csv')
    return redirect(url_for('home'))

@app.route('/download_pdf_report')
def download_pdf_report():
    batch_results = session.get('batch_results')
    analytics = session.get('analytics')
    chart_data = session.get('chart_data')
    if not batch_results or not analytics or not chart_data:
        return redirect(url_for('home'))
    # Generate pie chart image
    with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as tmpfile:
        labels = list(chart_data.keys())
        sizes = list(chart_data.values())
        colors = ['#4CAF50', '#FFC107', '#F44336']
        plt.figure(figsize=(4, 4))
        plt.pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%', startangle=140)
        plt.title('Sentiment Distribution')
        plt.tight_layout()
        plt.savefig(tmpfile.name)
        plt.close()
        chart_path = tmpfile.name
    # Prepare PDF
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font('Arial', 'B', 16)
    pdf.cell(0, 10, 'Product Review Sentiment Report', ln=True, align='C')
    pdf.ln(10)
    pdf.set_font('Arial', '', 12)
    pdf.cell(0, 10, f"Total Reviews: {analytics['total']}", ln=True)
    pdf.cell(0, 10, f"Positive: {analytics['positive_pct']}%", ln=True)
    pdf.cell(0, 10, f"Neutral: {analytics['neutral_pct']}%", ln=True)
    pdf.cell(0, 10, f"Negative: {analytics['negative_pct']}%", ln=True)
    pdf.cell(0, 10, f"Average Confidence: {analytics['avg_conf']}", ln=True)
    pdf.ln(5)
    pdf.image(chart_path, x=60, w=90)
    pdf.ln(10)
    # Add top 3 positive and negative reviews
    positives = [r for r in batch_results if r['label'].startswith('Positive')]
    negatives = [r for r in batch_results if r['label'].startswith('Negative')]
    pdf.set_font('Arial', 'B', 12)
    pdf.cell(0, 10, 'Top Positive Reviews:', ln=True)
    pdf.set_font('Arial', '', 11)
    for r in positives[:3]:
        pdf.multi_cell(0, 8, f"- {r['text']}")
    pdf.ln(2)
    pdf.set_font('Arial', 'B', 12)
    pdf.cell(0, 10, 'Top Negative Reviews:', ln=True)
    pdf.set_font('Arial', '', 11)
    for r in negatives[:3]:
        pdf.multi_cell(0, 8, f"- {r['text']}")
    # Output PDF
    with tempfile.NamedTemporaryFile(suffix='.pdf', delete=False) as pdf_file:
        pdf.output(pdf_file.name)
        pdf_file.seek(0)
        pdf_bytes = pdf_file.read()
    os.remove(chart_path)
    os.remove(pdf_file.name)
    return send_file(
        io.BytesIO(pdf_bytes),
        mimetype='application/pdf',
        as_attachment=True,
        download_name='sentiment_report.pdf'
    )

last_csv_report = None

if __name__ == '__main__':
    app.run(debug=True)
