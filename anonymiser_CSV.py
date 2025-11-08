import pandas as pd
import os
import re

DOSSIER = "data"
FICHIERS = [
    "information initiales.csv",
    "Mesures J0.csv",
    "Mesures J+6 semaines.csv",
    "Interventions.csv",
]

# candidats plausibles pour la colonne "sujet"
CANDIDATS = [
    "sujets", "sujet", "id", "identifiant", "participant", "nom", "prénom",
    "subject", "name"
]

def _clean_colname(s: str) -> str:
    s = str(s).strip().lower()
    s = s.replace("\ufeff", "")           # BOM éventuel
    s = re.sub(r"\s+", " ", s)            # espaces multiples -> simple
    s = s.replace("é", "e").replace("è", "e").replace("ê", "e").replace("à","a").replace("ç","c")
    return s

def trouver_colonne_sujets(df: pd.DataFrame) -> str | None:
    # mapping "nom net" -> vrai nom
    norm_map = { _clean_colname(c): c for c in df.columns }
    # priorité : correspondance exacte avec 'sujets'
    if "sujets" in norm_map:
        return norm_map["sujets"]
    # sinon, essaie la liste de candidats
    for cand in CANDIDATS:
        if cand in norm_map:
            return norm_map[cand]
    # sinon, prends la première colonne
    return df.columns[0] if len(df.columns) else None

def anonymiser_fichiers():
    os.makedirs(f"{DOSSIER}/anonymes", exist_ok=True)
    mapping = {}
    compteur = 1

    for fichier in FICHIERS:
        chemin = os.path.join(DOSSIER, fichier)
        if not os.path.exists(chemin):
            print(f"⚠️ Fichier introuvable : {chemin}")
            continue

        # lecture robuste (CSV ; et , ; latin1)
        df = pd.read_csv(chemin, sep=";", decimal=",", encoding="latin1")

        col_suj = trouver_colonne_sujets(df)
        if col_suj is None:
            print(f"❌ Impossible d'identifier la colonne sujets dans {fichier} (aucune colonne)")
            continue

        # normalise la colonne identifiant
        df[col_suj] = (
            df[col_suj]
            .astype(str)
            .str.replace("\ufeff", "", regex=False)
            .str.strip()
            .str.replace(r"\s+", " ", regex=True)
        )

        # construit le mapping
        for nom in df[col_suj].unique():
            if nom not in mapping:
                mapping[nom] = f"Sujet{compteur}"
                compteur += 1

        # applique le mapping
        df[col_suj] = df[col_suj].map(mapping)

        # renomme proprement la colonne en "Sujets" pour la suite du pipeline
        if col_suj != "Sujets":
            df.rename(columns={col_suj: "Sujets"}, inplace=True)

        # sauvegarde
        sortie = os.path.join(DOSSIER, "anonymes", fichier)
        df.to_csv(sortie, sep=";", decimal=",", index=False, encoding="latin1")
        print(f"✅ {fichier} anonymisé → {sortie} (colonne sujets: '{col_suj}' → 'Sujets')")

    print("\n🎉 Tous les fichiers ont été anonymisés (sans fichier de correspondance).")

if __name__ == "__main__":
    anonymiser_fichiers()
