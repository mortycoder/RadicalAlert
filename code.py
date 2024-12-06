import numpy as np
import pandas as pd
import nltk
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report

# Download necessary NLTK data
nltk.download('stopwords')
nltk.download('punkt')
from nltk.corpus import stopwords

# Sample dataset - In reality, you would need a much larger and more diverse dataset
# Example: '0' represents non-radical behavior, '1' represents risk of radicalization.
data = {
    'text': [
        'Join the cause to fight for freedom',
        'The government is corrupt and must be overthrown',
        'All religions should be respected equally',
        'We must protect our community from external threats',
        'Violence is never the answer to any problem',
        'This country has been infiltrated by foreign agents',
        'Unity is strength, and we must fight for justice',
        'Peace is the only way forward for humanity',
        'Radical ideas should not be tolerated',
        'We must act to defend our values and way of life'
    ],
    'label': [1, 1, 0, 0, 0, 1, 1, 0, 0, 1]  # Labels: 1 = At risk, 0 = Not at risk
}

# Convert data into a pandas DataFrame
df = pd.DataFrame(data)

# Preprocess the text: Tokenize, remove stopwords, and vectorize
stop_words = set(stopwords.words('english'))

def preprocess_text(text):
    # Tokenize and remove stopwords
    words = nltk.word_tokenize(text.lower())  # Tokenize and convert to lowercase
    words = [word for word in words if word.isalpha() and word not in stop_words]
    return ' '.join(words)

df['processed_text'] = df['text'].apply(preprocess_text)

# Convert the text into numerical features using TF-IDF Vectorizer
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(df['processed_text'])

# Labels (0 = not at risk, 1 = at risk)
y = df['label']

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Initialize a model (Naive Bayes classifier)
model = MultinomialNB()

# Train the model
model.fit(X_train, y_train)

# Predict on the test set
y_pred = model.predict(X_test)

# Evaluate the model performance
print(classification_report(y_test, y_pred))

# Example: Test the model with new text data
new_text = ['We must defend our country from the enemy', 'Peaceful protest is our right']
new_text_processed = [preprocess_text(text) for text in new_text]
new_text_vectorized = vectorizer.transform(new_text_processed)
predictions = model.predict(new_text_vectorized)

print("Predictions: ", predictions)  # 1 = At risk, 0 = Not at risk
