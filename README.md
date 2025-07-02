# Sentiment Analysis Web Application

A comprehensive sentiment analysis web application built with Flask that analyzes text sentiment using machine learning. The application can process both individual text inputs and batch CSV files, providing detailed analytics and visualizations.

## Features

- **Real-time Sentiment Analysis**: Analyze individual text inputs with confidence scores
- **Batch Processing**: Upload CSV files for bulk sentiment analysis
- **Interactive Dashboard**: Visual analytics with charts and sentiment distribution
- **Word Cloud Generation**: Create word clouds from analyzed text
- **PDF Reports**: Download detailed analysis reports
- **Responsive UI**: Modern, mobile-friendly interface with Bootstrap

## Sentiment Categories

- 😀 **Positive**: Upbeat, optimistic, or favorable sentiment
- 😐 **Neutral**: Balanced or objective sentiment
- 😞 **Negative**: Critical, pessimistic, or unfavorable sentiment

## Technology Stack

- **Backend**: Flask (Python)
- **Machine Learning**: Scikit-learn (Logistic Regression + TF-IDF Vectorization)
- **Frontend**: HTML5, CSS3, Bootstrap 5, JavaScript
- **Data Processing**: Pandas, NumPy
- **Visualization**: Matplotlib, WordCloud
- **PDF Generation**: FPDF

## Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/Harshad-Gore/sentiment-analysis.git
   cd sentiment-analysis
   ```

2. **Create a virtual environment**
   ```bash
   python -m venv venv
   ```

3. **Activate the virtual environment**
   - Windows:
     ```bash
     venv\Scripts\activate
     ```
   - macOS/Linux:
     ```bash
     source venv/bin/activate
     ```

4. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

## Usage

### Running the Application

1. **Start the Flask server**
   ```bash
   python app.py
   ```

2. **Open your browser** and navigate to `http://localhost:5000`

### Training the Model (Optional)

If you want to retrain the model with your own data:

1. **Prepare your dataset** in CSV format with columns:
   - `clean_text`: The text to analyze
   - `category`: Sentiment labels (-1 for negative, 0 for neutral, 1 for positive)

2. **Update the CSV filename** in `train.ipynb` if needed

3. **Run the training notebook**
   ```bash
   jupyter notebook train.ipynb
   ```

### Using the Web Interface

1. **Single Text Analysis**:
   - Enter text in the input field
   - Click "Analyze Sentiment"
   - View results with confidence score

2. **Batch Analysis**:
   - Upload a CSV file with a 'text' column
   - View analytics dashboard with charts
   - Download PDF report
   - Generate word cloud

## API Endpoints

- `GET /`: Main dashboard
- `POST /`: Process text analysis
- `POST /batch`: Handle CSV file uploads
- `GET /download_report`: Download PDF analysis report
- `GET /download_wordcloud`: Download word cloud image

## Model Performance

The sentiment analysis model uses:
- **Algorithm**: Logistic Regression
- **Feature Extraction**: TF-IDF Vectorization (max 5000 features)
- **Training Data**: Twitter sentiment dataset
- **Accuracy**: Typically 85-90% on test data

## File Structure

```
sentiment-analysis/
├── app.py                 # Main Flask application
├── train.ipynb           # Model training notebook
├── sentiment_model.pkl   # Trained sentiment model
├── tfidf_vectorizer.pkl  # TF-IDF vectorizer
├── Twitter_Data.csv      # Training dataset
├── templates/
│   └── index.html        # Main web interface
├── requirements.txt      # Python dependencies
├── README.md            # Project documentation
└── LICENSE              # License file
```

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

1. Fork the project
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## Contact

For any questions, suggestions, or collaboration opportunities:

- **GitHub**: [@Harshad-Gore](https://github.com/Harshad-Gore)
- **Discord**: `raybyte`

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- Thanks to the open-source community for the amazing libraries and tools
- Scikit-learn for machine learning capabilities
- Flask for the lightweight web framework
- Bootstrap for the responsive UI components

---

⭐ **Star this repo if you find it helpful!**
