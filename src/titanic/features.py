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
from sklearn.model_selection import StratifiedGroupKFold
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


def group_cv(X, y, n_splits=5, n_repeats=5):
    """Découpages de CV où un même ticket (famille, groupe de voyage) n'est jamais
    à la fois dans le train et la validation.

    Sans ça, la CV voit en validation des passagers dont la famille est dans le train :
    elle est trop optimiste, alors que le test Kaggle contient surtout des groupes inconnus.
    Renvoie une liste de n_splits × n_repeats couples (index train, index validation),
    à passer en cv= de cross_val_score.
    """
    return [split for r in range(n_repeats)
            for split in StratifiedGroupKFold(n_splits, shuffle=True, random_state=r)
            .split(X, y, groups=X["Ticket"])]


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


def family(X):
    """(SibSp, Parch) -> SibSp, Parch, FamilySize en 3 groupes : 0 = seul, 1 = 2 à 4, 2 = 5 et +.

    Taille = SibSp + Parch + 1 (le +1 = le passager).
    """
    X = X.copy()
    X["FamilySize"] = pd.cut(X["SibSp"] + X["Parch"] + 1, [0, 1, 4, 11], labels=False)
    return X


# --------------------------------------------------------------------------------------
# Survie du groupe : les autres passagers du même ticket ont-ils survécu ?
# --------------------------------------------------------------------------------------

class GroupSurvival(BaseEstimator, TransformerMixin):
    """Ticket -> GroupSurv : taux de survie des AUTRES passagers du même ticket vus au fit.

    0.5 si aucun autre membre du groupe n'est connu (passager seul, ou groupe absent du train).

    Piège de fuite : sur les lignes du train, il ne faut pas compter le passager lui-même
    (sinon on lui donne sa propre réponse). D'où fit_transform « leave-one-out » :
    pour une ligne du train, on retire sa propre survie de la somme du groupe.
    transform (validation, test) utilise tout le train, puisque ces passagers n'y sont pas.

    Ne pas évaluer cette feature avec group_cv : par construction, aucun passager de
    validation n'y a de membre de groupe dans le train, la feature y vaut toujours 0.5.
    Le test Kaggle ressemble à une CV stratifiée (36 % des passagers du test ont un ticket
    déjà vu dans le train, contre 34 % en validation stratifiée).
    """

    def fit(self, X, y):
        t = pd.Series(np.asarray(y, dtype=float), index=X["Ticket"].values)
        g = t.groupby(level=0)
        self.sum_, self.count_ = g.sum(), g.count()
        return self

    def transform(self, X):
        s = X["Ticket"].map(self.sum_).values
        n = X["Ticket"].map(self.count_).fillna(0).values
        out = np.where(n > 0, s / np.maximum(n, 1), 0.5)
        return out.reshape(-1, 1)

    def fit_transform(self, X, y):
        self.fit(X, y)
        s = X["Ticket"].map(self.sum_).values - np.asarray(y, dtype=float)
        n = X["Ticket"].map(self.count_).values - 1
        out = np.where(n > 0, s / np.maximum(n, 1), 0.5)
        return out.reshape(-1, 1)

    def get_feature_names_out(self, input_features=None):
        return np.array(["GroupSurv"])


def _ticket_parts(tickets):
    """Ticket -> (préfixe normalisé, numéro final). "STON/O2. 3101270" -> ("STONO2", 3101270).

    Ticket sans préfixe -> préfixe "" ; numéro non numérique (ex. "LINE") -> NaN.
    """
    parts = tickets.str.rsplit(" ", n=1)
    num = pd.to_numeric(parts.str[-1], errors="coerce").values
    pre = np.where(parts.str.len() > 1,
                   parts.str[0].str.upper().str.replace(r"[^A-Z0-9]", "", regex=True), "")
    return pre, num


def _family_key(X):
    """Nom de famille + classe + port d'embarquement (manquant = S), ex. "Jussila|3|S".

    La classe et le port évitent de réunir deux familles sans lien qui portent le même nom.
    """
    surname = X["Name"].str.split(",").str[0].str.strip()
    return (surname + "|" + X["Pclass"].astype(str) + "|" + X["Embarked"].fillna("S")).values


def group_rules(X_train, y_train, X_new, base_pred, max_ticket_gap=None):
    """Corrige des prédictions avec la survie du groupe vue dans le train.

    Groupe d'un passager = les passagers du train qui ont le même ticket. Avec
    max_ticket_gap=k, on y ajoute ceux qui ont le même nom de famille, la même classe,
    le même port, le même préfixe de ticket et un numéro de ticket à au plus k d'écart :
    une famille a parfois acheté des tickets qui se suivent (Jussila 4136 / 4137).
    k=1 est le seul écart validé sur le train (toutes les familles retrouvées y sont à 1) ;
    un écart plus grand relie aussi des homonymes sans lien (les Andersson).

    On ne regarde que les femmes et les garçons (titre Master) du groupe : le sort
    des hommes adultes ne dit rien (ils mouraient presque tous, quel que soit le groupe).
      - femme dont tous les femmes/garçons connus du groupe sont morts  -> morte (0)
      - garçon dont tous les femmes/garçons connus du groupe ont survécu -> survivant (1)
    Les autres passagers gardent base_pred.
    """
    title_tr = NameExtractor().transform(X_train[["Name"]]).ravel()
    wc = (X_train["Sex"].values == "female") | (title_tr == "Master")
    Xw = X_train[wc]

    # Membres de référence (femmes/garçons du train) et passagers à corriger, en positions 0..n-1
    pre_r, num_r = _ticket_parts(Xw["Ticket"])
    ref = pd.DataFrame({"ref": np.arange(len(Xw)), "Ticket": Xw["Ticket"].values,
                        "key": _family_key(Xw), "pre": pre_r, "num": num_r,
                        "surv": np.asarray(y_train, dtype=float)[wc]})
    pre_n, num_n = _ticket_parts(X_new["Ticket"])
    new = pd.DataFrame({"row": np.arange(len(X_new)), "Ticket": X_new["Ticket"].values,
                        "key": _family_key(X_new), "pre": pre_n, "num": num_n})

    # Liens (passager, membre de référence) : même ticket, puis éventuellement nom + ticket proche
    links = [new.merge(ref, on="Ticket")[["row", "ref", "surv"]]]
    if max_ticket_gap is not None:
        m = new.merge(ref, on="key", suffixes=("", "_r"))
        m = m[(m["pre"] == m["pre_r"]) & ((m["num"] - m["num_r"]).abs() <= max_ticket_gap)]
        links.append(m[["row", "ref", "surv"]])
    links = pd.concat(links).drop_duplicates(["row", "ref"])
    agg = links.groupby("row")["surv"].agg(["count", "sum"]).reindex(range(len(X_new)))
    n = agg["count"].fillna(0).values
    s = agg["sum"].fillna(0).values

    title_new = NameExtractor().transform(X_new[["Name"]]).ravel()
    pred = np.asarray(base_pred).copy()
    pred[(X_new["Sex"].values == "female") & (n > 0) & (s == 0)] = 0
    pred[(title_new == "Master") & (n > 0) & (s == n)] = 1
    return pred


def solo_female_rule(X_new, base_pred, fare_min=8, age_min=25):
    """Corrige des prédictions : femme seule en 3e classe avec prix par personne > fare_min
    ou âge > age_min -> morte (0). Les autres passagers gardent base_pred.

    Femme seule = 3e classe, ni famille à bord (SibSp + Parch = 0) ni autre passager sur son
    ticket (TICKET_COUNTS, train + test). Prix par personne = Fare / nb de passagers du ticket.
    Âge manquant : seule la condition sur le prix joue.

    Seuils tirés du notebook « Titanic WCG+XGBoost » de Chris Deotte (arbre CART).
    Fragile : 65 % de mortes parmi les condamnées sur le train, +0.67 pt en CV avec ces seuils
    fixes, mais +0.13 pt seulement quand un arbre apprend ses seuils dans chaque pli ;
    sur Kaggle, 9 justes sur 17 changements (0.79904 -> 0.80143).
    """
    n_ticket = X_new["Ticket"].map(TICKET_COUNTS).fillna(1).values
    solo = ((X_new["Sex"].values == "female") & (X_new["Pclass"].values == 3)
            & (X_new["SibSp"].values + X_new["Parch"].values == 0) & (n_ticket == 1))
    fare_pp = X_new["Fare"].values / n_ticket
    pred = np.asarray(base_pred).copy()
    pred[solo & ((fare_pp > fare_min) | (X_new["Age"].fillna(0).values > age_min))] = 0
    return pred


# --------------------------------------------------------------------------------------
# Le morceau de pipeline que les notebooks importent
# --------------------------------------------------------------------------------------

def make_column_transformer(name="title"):
    """Colonnes brutes -> colonnes numériques (avec des NaN, imputés plus loin dans le pipeline).

    name="title" : comme le pipeline RF_KNN ; name="tfidf" : comme le pipeline GBC.
    Age est rempli ici (médiane du titre) ; SibSp, Parch + FamilySize ; TicketGroup ;
    remainder="passthrough" laisse passer Pclass, Fare.
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
    ], remainder="passthrough")
    return ct
