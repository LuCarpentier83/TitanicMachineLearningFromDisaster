# Titanic – Machine Learning from Disaster

Compétition Kaggle d'initiation : prédire quels passagers du Titanic ont survécu
(`Survived` = 1) à partir de leurs informations (classe, sexe, âge, famille, ticket, prix…).
Le score Kaggle est l'**accuracy** (part de bonnes réponses) sur les 418 passagers du test :
chaque passager vaut 1/418 ≈ **0,24 point**.

**Meilleur score : 0.80143** (335 / 418), avec des règles simples (sexe + survie de la famille +
femmes seules de 3ᵉ classe), sans aucun modèle de machine learning.

---

## Installation

L'environnement Python est géré par [uv](https://docs.astral.sh/uv/) (versions exactes dans `uv.lock`).

```powershell
powershell -ExecutionPolicy Bypass -File .\install.ps1   # Windows
```
```bash
./install.sh                                             # Linux / macOS
```

Le script installe uv si besoin, crée `.venv` avec les dépendances, enregistre le kernel Jupyter
« Python (titanic) » et active les hooks git (`.githooks/commit-msg` refuse tout commit contenant
une attribution IA). Il est relançable sans risque.

Soumettre à Kaggle demande un jeton API Kaggle (`%USERPROFILE%\.kaggle\access_token` ou `kaggle.json`).

## Structure

| Chemin | Rôle |
|---|---|
| `src/titanic/features.py` | Tout le code partagé : nettoyage (morceaux de pipeline), CV groupée, survie du groupe, règles |
| `notebooks/02_random_forest.ipynb` | Random Forest |
| `notebooks/03_gradient_boosting.ipynb` | Gradient Boosting (GBC), CV groupée par ticket |
| `notebooks/04_survie_groupe.ipynb` | Survie du groupe (feature `GroupSurv` et règles femmes/garçons) |
| `scripts/submit.py` | Soumission Kaggle : vérifie le format, refuse un fichier déjà soumis, attend le score, note tout dans `submissions/history.csv` |
| `submissions/` | Fichiers soumis et historique des scores |
| `data/train.csv`, `data/test.csv` | Données Kaggle |
| `data/ntrain.csv`, `ntest.csv`, `submit.csv` | Anciens CSV pré-nettoyés (avant le passage aux pipelines), plus utilisés |

```powershell
uv run jupyter lab                                                    # notebooks
uv run python scripts/submit.py submissions/x.csv -m "description"    # soumettre
uv run python scripts/submit.py --list                                # dernières soumissions Kaggle
```

### Reproduire la meilleure soumission (0.80143)

```python
from titanic.features import load_data, group_rules, solo_female_rule, save_submission

X, y, test = load_data()
pred = (test.Sex == "female").astype(int).values           # 1. modèle « genre » : femmes vivent, hommes meurent
pred = group_rules(X, y, test, pred, max_ticket_gap=1)      # 2. corrections par la survie de la famille
pred = solo_female_rule(test, pred)                         # 3. femmes seules de 3e classe
save_submission(test, pred, "genre_regles_nom_femmes_seules")
```

---

## Les investigations, dans l'ordre

### Vocabulaire

- **CV (validation croisée)** : on coupe le train en 5 parts, on apprend sur 4 et on mesure sur la 5ᵉ, cinq fois,
  et on répète 5 fois (« 5 × 5 »). Sert à estimer le score sur des données jamais vues sans gaspiller de soumission.
- **CV stratifiée** : chaque part garde la même proportion de survivants que le train.
- **CV groupée par ticket** : un même ticket (une famille, un groupe de voyage) n'est jamais à la fois en apprentissage et en validation.
- **Surapprentissage** : le modèle apprend des détails propres au train qui ne se retrouvent pas ailleurs ; il brille en CV et chute sur Kaggle.
- **Fuite de données** : le modèle voit, pendant l'apprentissage, une information qu'il n'aurait pas en vrai (par exemple sa propre réponse).
- **Leave-one-out** : pour un passager du train, on calcule une statistique de son groupe *sans lui*, pour ne pas lui donner sa propre réponse.
- **RF / GBC** : Random Forest et Gradient Boosting, deux familles de modèles faits d'arbres de décision.

### 1. Modèles d'arbres (RF, GBC) — plafond vers 0.775

Nettoyage dans `features.py` : pont de la cabine (inconnue = pont « U »), port manquant = S,
titres extraits du nom (Mr, Mrs, Miss, Master, « Rare » qui ne contient que des hommes),
âge manquant = médiane du titre, puis RF et GBC.

| Version | CV | Kaggle |
|---|---|---|
| RF + KNN (premier modèle) | | 0.77511 |
| GBC titres, sans PassengerId | | 0.74641 |
| RF titres, sans PassengerId | | 0.75358 |
| GBC sans nombre de cabines | | 0.77511 |
| GBC pont U, port S, âge par titre, titres regroupés | | 0.77511 |
| GBC + FamilySize, TicketGroup, **FarePerPerson** | 0.8323 | **0.73684** |
| GBC + FamilySize, TicketGroup, Fare brut | 0.8310 | 0.76794 |

**Constat** : le GBC perd ~6 points entre la CV (0.83) et Kaggle (0.77), il surapprend.
Le piège **FarePerPerson** (prix divisé par le nombre de passagers du ticket) le montre : meilleure CV, pire score Kaggle.

### 2. CV groupée par ticket — n'aurait pas détecté le piège

Hypothèse : la CV stratifiée est trop optimiste parce que des membres d'une même famille se retrouvent
en apprentissage *et* en validation. On a donc mis en place une CV où un ticket n'est jamais coupé (`group_cv`).

- Scores plus bas, plus honnêtes : GBC 0.8193, RF 0.8063.
- Mais elle classait **toujours la version FarePerPerson en tête** : elle n'aurait pas évité le piège.

### 3. Survie du groupe — le vrai levier

Idée : sur le Titanic, les femmes et les enfants d'une même famille ont souvent vécu ou péri ensemble.

**Feature `GroupSurv`** (taux de survie des *autres* membres du même ticket, leave-one-out sur le train) :
seulement **+0,2 pt** dans le GBC. L'information est forte mais ne concerne que peu de passagers, et les arbres la noient.
À noter : la CV groupée est inadaptée à cette feature (0 % des validations ont un groupe connu,
contre 36 % du test Kaggle) ; il faut la CV stratifiée (34 %).

**Règles appliquées après le modèle** (`group_rules`), en ne regardant que les femmes et garçons (titre Master) du groupe :
- femme dont toutes les femmes/garçons connus du groupe sont morts → **meurt** ;
- garçon dont toutes les femmes/garçons connus du groupe ont survécu → **survit**.

| Version | CV strat. | Kaggle |
|---|---|---|
| Genre seul (femmes vivent, hommes meurent) | 0.7867 | 0.76555 |
| **Genre + règles** | 0.8368 | **0.79665** |
| GBC + règles | 0.8364 | 0.77272 |

- Genre + règles : 15 prédictions changées par rapport au genre seul, **14 justes**.
- GBC + règles : diffère de genre + règles sur 40 passagers, le GBC a raison 15 fois et **tort 25 fois**.
  Quand le GBC contredit la règle du sexe, il se trompe plus souvent qu'il ne tombe juste.
  La CV (0.8364 contre 0.8368) ne voyait pas la différence.

### 4. Groupes par nom de famille

Hypothèse : des familles ont des tickets différents ; grouper aussi par nom (+ classe + port) trouverait plus de groupes.

- **Nom seul** (union ticket ou nom) : 3 prédictions changées sur le test. Sur le train : 6 corrections gagnées, **6 perdues**.
  Les pertes viennent toutes d'un **faux groupe** : deux familles Andersson sans lien (tickets 347082 et 3101281).
- **Nom + tickets proches** (`max_ticket_gap`) : même nom, classe, port, préfixe de ticket et numéros à au plus *k* d'écart.
  Sur le train, toutes les vraies familles retrouvées sont à **1 d'écart** (Vander Planke 345763/4, Jussila 4136/7, Bourke 364848/9) :
  6 corrections, 6 justes, le faux groupe Andersson disparaît. Seul k = 1 est validé par le train ; au-delà
  on relie des cas non vérifiables (dont un autre Andersson à 9 d'écart).

| Version | CV strat. | Kaggle |
|---|---|---|
| Genre + règles, ticket | 0.8368 | 0.79665 |
| **Genre + règles, ticket + nom (écart ≤ 1)** | 0.8431 | **0.79904** |

Sur le test, une seule prédiction change (Ida Ilmakangas), et elle était juste. Le ticket capte déjà presque toutes les familles.

### 5. Ce qui se fait de mieux (notebooks de Chris Deotte)

Lus pour comparaison :
[Titanic using Name only](https://www.kaggle.com/code/cdeotte/titanic-using-name-only-0-81818) et
[Titanic WCG+XGBoost](https://www.kaggle.com/code/cdeotte/titanic-wcg-xgboost-0-84688).

- Son modèle **WCG** (woman-child-groups) applique les mêmes deux règles, avec des groupes par nom de famille
  (hors hommes adultes) complétés par le ticket.
- **Attention** : ses scores (0.818 → 0.847) étaient calculés sur **209 passagers** (la moitié du test) ;
  Kaggle note aujourd'hui les 418. Lui-même estimait le WCG à ~0.809 sur le test complet.
  Notre 0.799–0.801 est donc au niveau du WCG.
- Prédire quels **hommes adultes** survivent : motifs dans le train (1ʳᵉ classe, 20–36 ans), mais gain nul sur le test.
- Prédire quelles **femmes seules de 3ᵉ classe** meurent : son meilleur levier après le WCG.

### 6. Femmes seules de 3ᵉ classe

Règle (`solo_female_rule`, seuils d'un arbre CART de Deotte) : femme de 3ᵉ classe, sans famille à bord,
seule sur son ticket, avec un **prix par personne > 8 $ ou un âge > 25 ans → meurt**.

| Version | CV strat. | Kaggle |
|---|---|---|
| Groupes ticket + nom | 0.8431 | 0.79904 |
| **+ femmes seules, seuils fixes (8 $ / 25 ans)** | 0.8498 | **0.80143** |
| + femmes seules, arbre appris dans chaque pli (version honnête) | 0.8444 | non soumis |

- Train : 20 femmes condamnées, 13 vraiment mortes (65 %).
- Test : 17 changées, **9 justes et 8 fausses** (53 %), gain net d'une seule passagère.
- La version honnête (l'arbre doit trouver ses seuils sans voir la partie validation) ne gagne que +0,13 pt :
  les seuils 8 $ / 25 ans sont en partie taillés sur le train. Règle **fragile**, gardée parce qu'elle donne le record.

---

## Ce qu'on a appris

1. **Le sexe fait l'essentiel** (0.766). La famille ajoute ~3 points. Le reste se joue sur une poignée de passagers.
2. **Les modèles d'arbres surapprennent** sur ce petit jeu de données : leurs écarts à la règle du sexe sont surtout du bruit.
   Une règle simple + quelques corrections ciblées bat tous nos RF/GBC.
3. **La CV ne suffit pas à trier les bonnes idées** : elle n'a vu ni le piège FarePerPerson, ni la faiblesse de GBC + règles,
   et elle surestime les règles dont les seuils ont été choisis sur tout le train. Pour une règle, préférer une version
   « honnête » où les seuils sont réappris dans chaque pli.
4. **Chaque soumission se joue à quelques passagers** : un écart de 1 ou 2 bonnes réponses (0,24 à 0,5 pt) relève du hasard.
5. **Le plafond réaliste** sans tricher est vers 0.81–0.82 sur le test complet.

## Pistes restantes

- Aligner nos groupes sur le WCG de Deotte (nom de famille seul + rattachement par ticket) : il condamne 14 femmes, nous 8.
- Mettre à jour le notebook 04 pour utiliser `max_ticket_gap=1` et `solo_female_rule`.
- Ménage : même imputer pour RF et GBC, retirer le StandardScaler (inutile pour des arbres), colonne Age_manquant,
  traiter Fare = 0, supprimer le code mort de `features.py` (ticket_transformer, isNaN, rare_titles, docstrings périmées)
  et les anciens CSV de `data/`.
