import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import re

# Load data
df = pd.read_csv('dataset_reviews.csv')
df = df.dropna(subset=['content', 'sentiment'])

# Basic cleaning
def clean_text(text):
    text = text.lower()
    text = re.sub(r'[^a-z\s]', '', text)
    return text

df['cleaned'] = df['content'].apply(clean_text)

# We can balance the dataset or just use it.
# Let's try as-is.
X = df['cleaned']
y = df['sentiment']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
X_train2, X_test2, y_train2, y_test2 = train_test_split(X, y, test_size=0.3, random_state=42)

vectorizer = TfidfVectorizer(max_features=5000)
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

X_train2_vec = vectorizer.fit_transform(X_train2)
X_test2_vec = vectorizer.transform(X_test2)

print("--- SVM (80/20) ---")
svm = SVC(kernel='linear')
svm.fit(X_train_vec, y_train)
print("Train Acc:", accuracy_score(y_train, svm.predict(X_train_vec)))
print("Test Acc:", accuracy_score(y_test, svm.predict(X_test_vec)))

print("\n--- Random Forest (70/30) ---")
rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X_train2_vec, y_train2)
print("Train Acc:", accuracy_score(y_train2, rf.predict(X_train2_vec)))
print("Test Acc:", accuracy_score(y_test2, rf.predict(X_test2_vec)))
