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

DECKS = ["A", "B", "C", "D", "E", "F", "G", "T", "U"]  # 1re lettre de Cabin, U = inconnue
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
    deck = c[0] if isinstance(c, str) else "U"   # NaN n'est pas un str -> pont "U"
    return np.array([[1 * (v == deck) for v in DECKS]])


def embarked(p):
    port = p if isinstance(p, str) else "S"   # 2 manquantes (Icard, Stone) : port le plus fréquent, et le vrai
    return np.array([[1 * (v == port) for v in PORTS]])


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


cabin_transformer = Func_To_Transformer(cabin, ["Cabin"], ["Pont_" + v for v in DECKS])
ticket_transformer = Func_To_Transformer(ticket, ["Ticket"], ["Ticket"])
sex_transformer = Func_To_Transformer(sex, ["Sex"], ["Sex"])
embarked_transformer = Func_To_Transformer(embarked, ["Embarked"], ["Port_" + p for p in PORTS])


# --------------------------------------------------------------------------------------
# Name : deux encodages possibles
# --------------------------------------------------------------------------------------

# Titres de femmes ramenés vers Miss / Mrs, pour que Rare ne contienne que des hommes
# (sauf « Dr » : une seule femme, et le nom seul ne dit pas le sexe).
TITLE_MAP = {"Mlle": "Miss", "Ms": "Miss",
             "Mme": "Mrs", "Dona": "Mrs", "Lady": "Mrs", "Countess": "Mrs"}


class NameExtractor(BaseEstimator, TransformerMixin):
    """Name -> titre : Mr / Mrs / Miss / Master, sinon Rare (titres de femmes regroupés via TITLE_MAP)."""

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
            title = TITLE_MAP.get(title, title)
            title = title if title in ["Mr", "Mrs", "Miss", "Master"] else "Rare"
            titles.append(title)
        return np.array(titles).reshape(-1, 1)


def name_title():
    """Encodage du pipeline RF : titre extrait, puis one-hot."""
    return Pipeline([
        ("name_extractor", NameExtractor()),
        ("one_hot_encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
    ])


# --------------------------------------------------------------------------------------
# Age : âges manquants remplis par la médiane du titre
# --------------------------------------------------------------------------------------

class TitleMedianAge(BaseEstimator, TransformerMixin):
    """(Name, Age) -> Age, les âges manquants remplis par la médiane du titre.

    Les médianes sont apprises au fit, donc sur la partie entraînement de chaque fold
    seulement (pas de fuite). Remplace l'imputer pour l'âge : avec Ridge, il devinait
    des âges négatifs (ex. Sage, Master. Thomas Henry : -9.9 ans).
    """

    def fit(self, X, y=None):
        titles = NameExtractor().transform(X[["Name"]]).ravel()
        self.medians_ = pd.Series(X["Age"].values).groupby(titles).median().to_dict()
        self.global_ = X["Age"].median()   # secours si un titre n'a aucun âge connu
        return self

    def transform(self, X):
        titles = NameExtractor().transform(X[["Name"]]).ravel()
        age = X["Age"].values.astype(float)
        med = np.array([self.medians_.get(t, self.global_) for t in titles])
        return np.where(np.isnan(age), med, age).reshape(-1, 1)


# --------------------------------------------------------------------------------------
# Famille : taille de la famille à bord
# --------------------------------------------------------------------------------------

# Nombre de passagers par ticket, compté sur train + test.
# On ne lit que les numéros de ticket (jamais Survived) : pas de fuite.
_all = pd.concat([pd.read_csv(DATA / "train.csv"), pd.read_csv(DATA / "test.csv")])
TICKET_COUNTS = _all["Ticket"].value_counts()


def ticket_group(X):
    """Ticket -> TicketGroup en 3 groupes : 0 = seul sur son ticket, 1 = 2 à 4, 2 = 5 et +."""
    size = X["Ticket"].map(TICKET_COUNTS).fillna(1)
    return pd.DataFrame({"TicketGroup": pd.cut(size, [0, 1, 4, np.inf], labels=False)})


def fare_per_person(X):
    """(Ticket, Fare) -> FarePerPerson = Fare ÷ nombre de passagers sur le ticket.

    Fare est le prix du ticket entier : 69.55 pour les 11 Sage, soit 6.32 par personne.
    """
    size = X["Ticket"].map(TICKET_COUNTS).fillna(1)
    return pd.DataFrame({"FarePerPerson": X["Fare"] / size})


def family(X):
    """(SibSp, Parch) -> SibSp, Parch, FamilySize en 3 groupes : 0 = seul, 1 = 2 à 4, 2 = 5 et +.

    Taille = SibSp + Parch + 1 (le +1 = le passager).
    """
    X = X.copy()
    X["FamilySize"] = pd.cut(X["SibSp"] + X["Parch"] + 1, [0, 1, 4, 11], labels=False)
    return X


# --------------------------------------------------------------------------------------
# Le morceau de pipeline que les notebooks importent
# --------------------------------------------------------------------------------------

def make_column_transformer(name="title"):
    """Colonnes brutes -> colonnes numériques (avec des NaN, imputés plus loin dans le pipeline).

    name="title" : comme le pipeline RF_KNN ; name="tfidf" : comme le pipeline GBC.
    Age est rempli ici (médiane du titre) ; SibSp, Parch + FamilySize ; TicketGroup ; FarePerPerson
    à la place de Fare ; remainder="passthrough" laisse passer Pclass.
    """
    if name == "title":
        name_step = ("name", name_title(), ["Name"])
    else:
        raise ValueError(f"name doit valoir 'title' ou 'tfidf', pas {name!r}")

    ct = ColumnTransformer(transformers=[
        ("drop_id", "drop", ["PassengerId"]),
        ("cabin", cabin_transformer.get_transformer, cabin_transformer.cols_in),
        ("sex", sex_transformer.get_transformer, sex_transformer.cols_in),
        ("embarked", embarked_transformer.get_transformer, embarked_transformer.cols_in),
        name_step,
        ("age", TitleMedianAge(), ["Name", "Age"]),
        ("family", FunctionTransformer(family), ["SibSp", "Parch"]),
        ("ticket_group", FunctionTransformer(ticket_group), ["Ticket"]),
        ("fare", FunctionTransformer(fare_per_person), ["Ticket", "Fare"]),
    ], remainder="passthrough")
    return ct
