from flask import Flask, request, jsonify
import pickle
from flask.templating import render_template
from pandas._libs.tslibs import vectorized
from scipy.sparse import data

app = Flask(__name__)

with open("models/Sentiment_model.pkl", "rb") as model_file:
    model = pickle.load(model_file)

with open("models/Tfidf_vectorizer.pkl", "rb") as vectorizer_file:
    vectorizer = pickle.load(vectorizer_file)

    @app.route('/')
    def home():
        return render_template('index.html')

@app.route('/predict', methods=['POST'])

def predict():
    text = request.form.get('text', '')

    if not text:
        return render_template('index.html', error='Por favor, ingresa un texto válido.'), 400

    vectorized_text = vectorizer.transform([text])
    prediction = model.predict(vectorized_text) [0]

    return render_template('index.html', prediction=prediction, text=text)

if __name__ == '__main__':
    app.run(debug=True)
