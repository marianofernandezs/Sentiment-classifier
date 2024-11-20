from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score, f1_score
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

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.3, random_state=42 )

def compare_models(X_train, X_test, Y_train, Y_test):

    print("Evaluación De Los Modelos Avanzados\n")

    # Modelo RF
    rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
    rf_model.fit(X_train, Y_train)
    accuracy_rf = accuracy_score(Y_test, rf_model.predict(X_test))
    roc_auc_rf = roc_auc_score(Y_test, rf_model.predict_proba(X_test)[:, 1]) #type: ignore
    f1_rf = f1_score(Y_test, rf_model.predict(X_test), pos_label='positive') #type: ignore
    print("\nModelo: RF")
    print(f"Precisión: {accuracy_rf:.3f}")
    print(f"F1-Score: {f1_rf:.3f} ")
    print(f"ROC AUC Score: {roc_auc_rf:.3f}")


    # Regresión Logística
    lr_model = LogisticRegression(max_iter=250)
    lr_model.fit(X_train, Y_train)
    accuracy_lr = accuracy_score(Y_test, lr_model.predict(X_test))
    roc_auc_lr = roc_auc_score(Y_test, lr_model.predict_proba(X_test)[:, 1])
    f1_lr = f1_score(Y_test, lr_model.predict(X_test), pos_label='positive') #type: ignore
    print("\nModelo: Logistic Regression")
    print(f"Precisión: {accuracy_lr:.3f}")
    print(f"F1-Score: {f1_lr:.3f} ")
    print(f"ROC AUC Score: {roc_auc_lr:.3f}")


    # MLP
    mlp_model = MLPClassifier(hidden_layer_sizes=(40, ), max_iter=350)
    mlp_model.fit(X_train, Y_train)
    accuracy_mlp = accuracy_score(Y_test, mlp_model.predict(X_test))
    roc_auc_mlp = roc_auc_score(Y_test, mlp_model.predict_proba(X_test)[:, 1]) #type: ignore
    f1_mlp = f1_score(Y_test, mlp_model.predict(X_test), pos_label='positive') #type: ignore
    print("\nModelo: MLP")
    print(f"Precisión: {accuracy_mlp:.3f}")
    print(f"F1-Score: {f1_mlp:.3f} ")
    print(f"ROC AUC Score: {roc_auc_mlp:.3f}")

    results= { "Modelo": ["Random Forest", "Logistic Regression", "MLP"],
            "Precisión": [accuracy_rf, accuracy_lr, accuracy_mlp],
            "F1-Score": [f1_rf, f1_lr, f1_mlp],
            "ROC AUC": [roc_auc_rf, roc_auc_lr, roc_auc_mlp],
        }

    df_results = pd.DataFrame(results)
    print("\nResumen De Resultados:\n",df_results)

    df_results.to_csv("Model_Comparision_Results.csv", index=False)
    print("Resultados guardados en 'model_comparison_results.csv'.")

compare_models(X_train, X_test, Y_train, Y_test)
