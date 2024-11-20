import pandas as pd
import re
from nltk.corpus import stopwords
from nltk.tag.api import accuracy
from scipy.sparse import data
from nltk.tokenize import RegexpTokenizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, roc_auc_score
import nltk
import pickle


nltk.download('stopwords')
nltk.download('omw-1.4')

data = pd.read_csv("../data/dataset.csv")

tokenizer = RegexpTokenizer(r'\w+')
stop_words = set(stopwords.words("english"))

def clean_text(text):
    text = text.lower()
    text = re.sub(r"<.*?>"," ",text)
    text = re.sub(r"[^a-z\s]"," ",text)

    words = tokenizer.tokenize(text)

    words = [word for word in words if word not in stop_words]
    return " ".join(words)


data["cleaned_review"] = data["review"].apply(clean_text)


X = data['cleaned_review']
Y = data['sentiment']

vectorizer = TfidfVectorizer(max_features=5000)

X = vectorizer.fit_transform(X).toarray() #type: ignore

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42 )

model = LogisticRegression(max_iter=10000)
model.fit(X_train, Y_train)

y_pred = model.predict(X_test)

accuracy = accuracy_score(Y_test, y_pred)
print("Precisión del modelo:", accuracy)

conf_matrix = confusion_matrix(Y_test, y_pred)
print("Matriz de confusión:\n", conf_matrix)

report = classification_report(Y_test, y_pred)
print("Reporte de clasificación:\n", report)

y_prob = model.predict_proba(X_test)[:,1]

auc_score = roc_auc_score(Y_test, y_prob)
print(f"ROC AUC Score: {auc_score:.2f}")

with open("../models/Sentiment_model.pkl","wb") as model_file:
    pickle.dump(model, model_file)
with open("../models/Tfidf_vectorizer.pkl", "wb") as vectorizer_file:
    pickle.dump(vectorizer, vectorizer_file)
with open("Model_evalution.txt", "w") as f:
    f.write(f"Precisión del modelo: {accuracy:.2f}\n")
    f.write(f"ROC AUC Score: {auc_score:.2f}\n")
    f.write(f"Matriz de confusión:\n{conf_matrix}\n")
    f.write(f"Reporte de clasificación:\n{report}\n")
print("Resultados guardados en 'Model_evalution.txt'.")

print(f"Modelo entrenado con éxito. Precisión: {accuracy:.2f}")

print("Modelo y Vectorizador Guardados.")
