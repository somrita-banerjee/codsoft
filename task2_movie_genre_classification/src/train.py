import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score
import joblib
from preprocess import clean_text

# Load the dataset
df = pd.read_csv("data/genre-classification-dataset.csv")  # Modify filename as needed

# Preprocess
df['clean_plot'] = df['plot'].apply(clean_text)

# Vectorize
vectorizer = TfidfVectorizer(max_features=5000)
X = vectorizer.fit_transform(df['clean_plot'])
y = df['genre']

# Split and train
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = LogisticRegression(max_iter=300)
model.fit(X_train, y_train)

# Evaluate
y_pred = model.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred))

# Save outputs
joblib.dump(model, "models/movie_genre_model.pkl")
joblib.dump(vectorizer, "models/tfidf_vectorizer.pkl")
