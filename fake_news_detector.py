# Part 1: Imports
import pandas as pd              # For data manipulation and analysis
import numpy as np              # For numerical operations
from sklearn.model_selection import train_test_split  # For splitting dataset
from sklearn.feature_extraction.text import TfidfVectorizer  # For text vectorization
from sklearn.svm import LinearSVC  # The classifier we're using
import joblib                   # For saving/loading trained models
import os                       # For file and system operations
import time                     # For adding delays if needed

# Part 2: The FakeNewsDetector Class
class FakeNewsDetector:
    def __init__(self):
        # Initialize empty vectorizer and classifier
        self.vectorizer = None
        self.classifier = None
    
    def train_model(self):
        """Trains the model using the dataset"""
        # Load dataset
        data = pd.read_csv('fake_or_real_news.csv')
        
        # Convert labels to binary (0 for REAL, 1 for FAKE)
        data['fake'] = data['label'].apply(lambda x: 0 if x == "REAL" else 1)
        data = data.drop("label", axis=1)  # Remove original label column
        
        # Separate features (text) and labels (fake/real)
        X, y = data['text'], data['fake']
        
        # Split into training and testing sets (80% train, 20% test)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
        
        # Create and train the vectorizer
        # stop_words='english' removes common English words
        # max_df=0.7 ignores terms that appear in >70% of documents
        self.vectorizer = TfidfVectorizer(stop_words='english', max_df=0.7)
        X_train_vectorized = self.vectorizer.fit_transform(X_train)
        
        # Train the classifier
        self.classifier = LinearSVC()
        self.classifier.fit(X_train_vectorized, y_train)
        
        # Save the trained models
        joblib.dump(self.vectorizer, 'vectorizer.joblib')
        joblib.dump(self.classifier, 'classifier.joblib')
    
    def load_model(self):
        """Loads previously trained model"""
        self.vectorizer = joblib.load('vectorizer.joblib')
        self.classifier = joblib.load('classifier.joblib')
    
    def predict(self, text):
        """Makes prediction on new text"""
        # Transform text using the trained vectorizer
        vectorized_text = self.vectorizer.transform([text])
        # Make prediction
        prediction = self.classifier.predict(vectorized_text)[0]
        return "REAL" if prediction == 0 else "FAKE"

# Part 3: Utility Functions
def create_input_file():
    """Creates input file if it doesn't exist"""
    if not os.path.exists("input_news.txt"):
        with open("input_news.txt", "w", encoding="utf-8") as f:
            f.write("Paste your news article here")

def clear_screen():
    """Clears the console screen"""
    os.system('cls' if os.name == 'nt' else 'clear')

# Part 4: Main Function
def main():
    # Initialize detector
    detector = FakeNewsDetector()
    
    # Train or load model
    if not (os.path.exists('vectorizer.joblib') and os.path.exists('classifier.joblib')):
        print("Training new model...")
        detector.train_model()
        print("Model training complete!")
    else:
        print("Loading existing model...")
        detector.load_model()
        print("Model loaded successfully!")
    
    # Create input file
    create_input_file()
    
    # Main program loop
    while True:
        clear_screen()
        print("\nFake News Detector Menu:")
        print("1. Analyze news from input_news.txt")
        print("2. Open input_news.txt for editing")
        print("3. Exit")
        
        choice = input("\nEnter your choice (1-3): ")
        
        if choice == '1':
            try:
                # Read the input file
                with open("input_news.txt", "r", encoding="utf-8") as f:
                    news_text = f.read().strip()
                
                # Check if file is empty or contains default text
                if not news_text or news_text == "Paste your news article here":
                    print("\nThe input file is empty or contains default text.")
                    print("Please choose option 2 to edit the file first.")
                    input("\nPress Enter to continue...")
                    continue
                
                # Show preview of text being analyzed
                print("\nAnalyzing the following text:")
                print("-" * 50)
                print(news_text[:200] + "..." if len(news_text) > 200 else news_text)
                print("-" * 50)
                
                # Make prediction
                result = detector.predict(news_text)
                
                # Save and display result
                with open("result.txt", "w", encoding="utf-8") as f:
                    f.write(f"PREDICTION: This news article appears to be {result}\n\n")
                    f.write("ANALYZED TEXT:\n")
                    f.write(news_text[:200] + "..." if len(news_text) > 200 else news_text)
                
                print(f"\nAnalysis complete! The news article appears to be: {result}")
                print("Full details saved in result.txt")
                input("\nPress Enter to continue...")
                
            except FileNotFoundError:
                print("Error: input_news.txt not found!")
                input("\nPress Enter to continue...")
            except Exception as e:
                print(f"An error occurred: {str(e)}")
                input("\nPress Enter to continue...")
                
        elif choice == '2':
            # Open file for editing
            print("\nOpening input_news.txt for editing...")
            print("Please edit the file, save it, and close the editor.")
            
            if os.name == 'nt':  # Windows
                os.system('notepad input_news.txt')
            else:  # Unix-like
                editor = os.getenv('EDITOR', 'nano')
                os.system(f'{editor} input_news.txt')
            
            print("\nFile editing complete.")
            input("Press Enter to continue...")
            
        elif choice == '3':
            print("\nThank you for using Fake News Detector!")
            break
        else:
            print("\nInvalid choice! Please try again.")
            input("Press Enter to continue...")

# Part 5: Program Entry Point
if __name__ == "__main__":
    main()
    