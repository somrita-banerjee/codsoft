import joblib
from preprocess import clean_text

model = joblib.load("models/movie_genre_model.pkl")
vectorizer = joblib.load("models/tfidf_vectorizer.pkl")

def predict_genre(plot: str) -> str:
    cleaned = clean_text(plot)
    vect = vectorizer.transform([cleaned])
    return model.predict(vect)[0]

if __name__ == "__main__":
    example = "A brave hero battles inner demons to save his city."
    print("Predicted Genre:", predict_genre(example))
