"""Nettoyage des données Titanic, partagé par tous les notebooks de modèles.

Ce fichier reprend TEL QUEL le code de final_notebook_cleaning_column_transformer.ipynb
(y compris ses défauts connus, corrigés plus tard) : il ne fait que le ranger.

Règle : ici on écrit des MORCEAUX DE PIPELINE, jamais un CSV pré-nettoyé.
Le scaler et l'imputer restent dans le pipeline du notebook, pour être réappris à chaque
fold de la cross-validation (sinon fuite de données).
"""

import re
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder

# --------------------------------------------------------------------------------------
# Chemins et chargement
# --------------------------------------------------------------------------------------

# Racine du projet = dossier qui contient pyproject.toml
ROOT = next(p for p in Path(__file__).resolve().parents if (p / "pyproject.toml").exists())
DATA = ROOT / "data"
SUBMISSIONS = ROOT / "submissions"


def load_data():
    """Renvoie X (features du train), y (Survived) et test (features du test Kaggle)."""
    train = pd.read_csv(DATA / "train.csv")
    test = pd.read_csv(DATA / "test.csv")
    return train.drop(columns="Survived"), train["Survived"], test


def save_submission(test, ypred, name):
    """Écrit submissions/<name>.csv au format Kaggle et renvoie son chemin."""
    SUBMISSIONS.mkdir(exist_ok=True)
    path = SUBMISSIONS / f"{name}.csv"
    pd.DataFrame({"PassengerId": test["PassengerId"], "Survived": ypred}).to_csv(path, index=False)
    return path


# --------------------------------------------------------------------------------------
# Valeurs possibles (avant : calculées dans le notebook depuis X_train)
# --------------------------------------------------------------------------------------

DECKS = ["A", "B", "C", "D", "E", "F", "G", "T"]  # 1re lettre de Cabin
PORTS = ["C", "Q", "S"]                            # Embarked


# --------------------------------------------------------------------------------------
# Fonctions ligne par ligne (une valeur en entrée -> une ou plusieurs colonnes en sortie)
# --------------------------------------------------------------------------------------

def isNaN(x):
    return x != x


def sex(s):
    return 1 * (s == "male")


def ticket(t):
    try:
        return int(t.split()[-1])
    except Exception:
        return np.nan


def cabin(c):
    # Défaut connu : "C23 C25" ou "F G73" -> int() plante -> tout NaN, pont compris
    try:
        return np.array([[1 * (v == c[0]) for v in DECKS] + [int(c[1:])]])
    except Exception:
        return np.array([[np.nan for _ in range(len(DECKS) + 1)]])


def embarked(p):
    try:
        return np.array([[1 * (v == p[0]) for v in PORTS]])
    except Exception:
        return np.array([[np.nan for _ in range(len(PORTS))]])


# --------------------------------------------------------------------------------------
# Transformation d'une fonction ligne par ligne en transformer sklearn
# --------------------------------------------------------------------------------------

class CustomFunctionTransformer(FunctionTransformer):
    def __init__(self, func, cols_in, cols_out):
        FunctionTransformer.__init__(self, func)
        self.cols_in, self.cols_out = cols_in, cols_out


class Func_To_Transformer:
    def __init__(self, func, cols_in, cols_out):
        self.func = func
        self.size_in, self.size_out = len(cols_in), len(cols_out)
        self.cols_in, self.cols_out = cols_in, cols_out

    @property
    def get_transformer(self):
        def out_func(data):
            if isinstance(data, (pd.DataFrame, pd.Series)):
                data = data.values
            if data.ndim != 2 or data.shape[1] != self.size_out:
                data = data.reshape(data.shape[0], self.size_in)
            transformed_data = np.array([self.func(row[0]) for row in data])
            return transformed_data.reshape(data.shape[0], self.size_out)

        return CustomFunctionTransformer(out_func, self.cols_in, self.cols_out)


cabin_transformer = Func_To_Transformer(cabin, ["Cabin"], ["Pont_" + v for v in DECKS] + ["NumCab"])
ticket_transformer = Func_To_Transformer(ticket, ["Ticket"], ["Ticket"])
sex_transformer = Func_To_Transformer(sex, ["Sex"], ["Sex"])
embarked_transformer = Func_To_Transformer(embarked, ["Embarked"], ["Port_" + p for p in PORTS])


# --------------------------------------------------------------------------------------
# Name : deux encodages possibles
# --------------------------------------------------------------------------------------

class NameExtractor(BaseEstimator, TransformerMixin):
    """Name -> titre : Mr / Mrs / Miss / Master, sinon Rare."""

    def __init__(self, rare_titles=None):
        self.rare_titles = rare_titles if rare_titles else [
            "Dr", "Rev", "Mlle", "Major", "Col", "Countess", "Capt", "Ms", "Sir", "Lady", "Mme", "Don", "Jonkheer"]

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        if isinstance(X, pd.DataFrame):
            X = X.iloc[:, 0].values
        titles = []
        for name in X:
            match = re.search(r" ([A-Za-z]+)\.", name)
            title = match.group(1) if match else "Unknown"
            title = title if title in ["Mr", "Mrs", "Miss", "Master"] else "Rare"
            titles.append(title)
        return np.array(titles).reshape(-1, 1)


def name_title():
    """Encodage du pipeline RF : titre extrait, puis one-hot."""
    return Pipeline([
        ("name_extractor", NameExtractor()),
        ("one_hot_encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
    ])


def name_tfidf():
    """Encodage du pipeline GBC : TF-IDF sur les mots du nom (défaut connu : apprend des prénoms)."""
    return Pipeline([
        ("tfidf", TfidfVectorizer(max_features=100)),
        ("to_dense", FunctionTransformer(lambda x: x.todense())),
    ])


# --------------------------------------------------------------------------------------
# Le morceau de pipeline que les notebooks importent
# --------------------------------------------------------------------------------------

def make_column_transformer(name="title"):
    """Colonnes brutes -> colonnes numériques (avec des NaN, imputés plus loin dans le pipeline).

    name="title" : comme le pipeline RF_KNN ; name="tfidf" : comme le pipeline GBC.
    Défaut connu : remainder="passthrough" laisse passer PassengerId, Pclass, Age, SibSp, Parch, Fare.
    """
    if name == "title":
        name_step = ("name", name_title(), ["Name"])
    elif name == "tfidf":
        name_step = ("tdidf", name_tfidf(), "Name")
    else:
        raise ValueError(f"name doit valoir 'title' ou 'tfidf', pas {name!r}")

    ct = ColumnTransformer(transformers=[
        ("cabin", cabin_transformer.get_transformer, cabin_transformer.cols_in),
        ("ticket", ticket_transformer.get_transformer, ticket_transformer.cols_in),
        ("sex", sex_transformer.get_transformer, sex_transformer.cols_in),
        ("embarked", embarked_transformer.get_transformer, embarked_transformer.cols_in),
        name_step,
    ], remainder="passthrough")

    if name == "tfidf":
        # le TF-IDF renvoie une np.matrix : on la repasse en tableau numpy
        return Pipeline([("column_tranformer", ct), ("to_array", FunctionTransformer(lambda x: x.A))])
    return ct
