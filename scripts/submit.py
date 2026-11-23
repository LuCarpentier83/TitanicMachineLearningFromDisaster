"""Soumet un fichier de prédictions à Kaggle (Titanic) et affiche le score.

Usage (depuis la racine du projet, Linux ou Windows) :
    uv run python scripts/submit.py submissions/gbc.csv -m "GBC titre"
    uv run python scripts/submit.py --list            # historique des soumissions

Avant d'envoyer, le script :
  - vérifie le format (418 lignes, colonnes PassengerId / Survived, valeurs 0/1) ;
  - refuse un fichier identique à un fichier déjà soumis (submissions/history.csv), sauf --force ;
  - demande confirmation (o/N), sauf --yes.
Chaque soumission est notée dans submissions/history.csv (date, fichier, empreinte, message, score).
"""

import argparse
import csv
import hashlib
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

import pandas as pd

COMPETITION = "titanic"
ROOT = Path(__file__).resolve().parents[1]
HISTORY = ROOT / "submissions" / "history.csv"
HISTORY_COLS = ["date", "file", "sha256", "message", "public_score"]


def kaggle(*args):
    """Lance la CLI kaggle et renvoie sa sortie texte (arrête le script si elle échoue)."""
    res = subprocess.run([sys.executable, "-m", "kaggle", *args], capture_output=True, text=True)
    if res.returncode != 0:
        sys.exit(f"Erreur kaggle {' '.join(args)} :\n{res.stdout}{res.stderr}")
    return res.stdout


def list_submissions():
    rows = list(csv.DictReader(kaggle("competitions", "submissions", COMPETITION, "--csv").splitlines()))
    return [r for r in rows if r.get("ref")]


def check_format(path):
    df = pd.read_csv(path)
    test = pd.read_csv(ROOT / "data" / "test.csv")
    problems = []
    if list(df.columns) != ["PassengerId", "Survived"]:
        problems.append(f"colonnes {list(df.columns)} au lieu de ['PassengerId', 'Survived']")
    elif len(df) != len(test):
        problems.append(f"{len(df)} lignes au lieu de {len(test)}")
    elif set(df["PassengerId"]) != set(test["PassengerId"]):
        problems.append("les PassengerId ne correspondent pas à data/test.csv")
    elif not set(df["Survived"].unique()) <= {0, 1}:
        problems.append(f"Survived contient autre chose que 0/1 : {sorted(df['Survived'].unique())}")
    if problems:
        sys.exit("Fichier invalide : " + "; ".join(problems))
    return df["Survived"].mean()


def read_history():
    if not HISTORY.exists():
        return []
    with HISTORY.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def write_history(rows):
    HISTORY.parent.mkdir(exist_ok=True)
    with HISTORY.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=HISTORY_COLS)
        w.writeheader()
        w.writerows(rows)


def wait_for_score(ref, timeout=180):
    start = time.time()
    while time.time() - start < timeout:
        for r in list_submissions():
            if r["ref"] == ref:
                if "COMPLETE" in r["status"]:
                    return r["publicScore"]
                if "ERROR" in r["status"]:
                    sys.exit(f"Kaggle a rejeté la soumission : {r}")
        time.sleep(5)
    return None


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("file", nargs="?", help="fichier CSV à soumettre, ex. submissions/rf.csv")
    p.add_argument("-m", "--message", help="description de la soumission (obligatoire pour soumettre)")
    p.add_argument("--list", action="store_true", help="affiche les dernières soumissions et quitte")
    p.add_argument("--yes", action="store_true", help="ne pas demander de confirmation")
    p.add_argument("--force", action="store_true", help="soumettre même si ce fichier a déjà été soumis")
    a = p.parse_args()

    if a.list:
        for r in list_submissions()[:15]:
            print(f"{r['date'][:16]}  {r['publicScore'] or '-':>8}  {r['fileName']:<15} {r['description']}")
        return
    if not a.file or not a.message:
        p.error("il faut un fichier et -m \"message\" (ou --list)")

    path = Path(a.file).resolve()
    if not path.exists():
        sys.exit(f"Fichier introuvable : {path}")
    survival = check_format(path)
    digest = hashlib.sha256(path.read_bytes()).hexdigest()

    history = read_history()
    same = [h for h in history if h["sha256"] == digest]
    if same and not a.force:
        h = same[-1]
        sys.exit(f"Fichier identique déjà soumis le {h['date']} ({h['message']}) : score {h['public_score']}.\n"
                 "Rien envoyé. Ajoute --force pour le soumettre quand même.")

    print(f"Fichier  : {path.relative_to(ROOT) if path.is_relative_to(ROOT) else path}")
    print(f"Message  : {a.message}")
    print(f"Survie prédite : {survival:.1%}")
    if not a.yes and input("Soumettre à Kaggle ? [o/N] ").strip().lower() not in ("o", "oui", "y", "yes"):
        sys.exit("Annulé, rien envoyé.")

    kaggle("competitions", "submit", COMPETITION, "-f", str(path), "-m", a.message)
    ref = list_submissions()[0]["ref"]  # la plus récente = la nôtre
    print("Envoyé, attente du score…")
    score = wait_for_score(ref)
    print(f"Score public : {score}" if score else "Score pas encore disponible : relance avec --list plus tard.")

    history.append({"date": datetime.now().strftime("%Y-%m-%d %H:%M"), "file": path.name,
                    "sha256": digest, "message": a.message, "public_score": score or ""})
    write_history(history)


if __name__ == "__main__":
    main()
