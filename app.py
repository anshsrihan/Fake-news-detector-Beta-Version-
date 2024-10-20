from flask import Flask, render_template, request, jsonify
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
import joblib
import os

app = Flask(__name__)

class FakeNewsDetector:
    def __init__(self):
        self.vectorizer = None
        self.classifier = None
            
    def train_model(self):
        # Load and prepare data
        data = pd.read_csv('fake_or_real_news.csv')
        data['fake'] = data['label'].apply(lambda x: 0 if x == "REAL" else 1)
        data = data.drop("label", axis=1)
        
        X, y = data['text'], data['fake']
        
        # Create and train the vectorizer
        self.vectorizer = TfidfVectorizer(stop_words='english', max_df=0.7)
        X_train_vectorized = self.vectorizer.fit_transform(X)
        
        # Train the classifier
        self.classifier = LinearSVC()
        self.classifier.fit(X_train_vectorized, y)
        
        # Save the trained models
        joblib.dump(self.vectorizer, 'vectorizer.joblib')
        joblib.dump(self.classifier, 'classifier.joblib')
    
    def load_model(self):
        if os.path.exists('vectorizer.joblib') and os.path.exists('classifier.joblib'):
            self.vectorizer = joblib.load('vectorizer.joblib')
            self.classifier = joblib.load('classifier.joblib')
            return True
        return False
    
    def predict(self, text):
        vectorized_text = self.vectorizer.transform([text])
        prediction = self.classifier.predict(vectorized_text)[0]
        return "REAL" if prediction == 0 else "FAKE"

# Initialize detector
detector = FakeNewsDetector()

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/analyze', methods=['POST'])
def analyze():
    try:
        if not detector.vectorizer or not detector.classifier:
            if not detector.load_model():
                detector.train_model()

        text = request.form.get('news_text', '')
        if not text:
            return jsonify({
                'error': 'Please provide news text to analyze.'
            }), 400

        result = detector.predict(text)
        confidence = "High"  # You can add actual confidence calculation later
        
        return jsonify({
            'result': result,
            'confidence': confidence,
            'text_preview': text[:200] + '...' if len(text) > 200 else text
        })

    except Exception as e:
        return jsonify({
            'error': f'An error occurred: {str(e)}'
        }), 500

@app.route('/analyze-file', methods=['POST'])
def analyze_file():
    try:
        if 'file' not in request.files:
            return jsonify({
                'error': 'No file uploaded'
            }), 400

        file = request.files['file']
        if file.filename == '':
            return jsonify({
                'error': 'No file selected'
            }), 400

        text = file.read().decode('utf-8')
        result = detector.predict(text)
        confidence = "High"  # You can add actual confidence calculation later

        return jsonify({
            'result': result,
            'confidence': confidence,
            'text_preview': text[:200] + '...' if len(text) > 200 else text
        })

    except Exception as e:
        return jsonify({
            'error': f'An error occurred: {str(e)}'
        }), 500

if __name__ == '__main__':
    app.run(debug=True)