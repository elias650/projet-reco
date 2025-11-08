# 🧠 Recommandation d’intervention de renforcement musculaire (6 semaines)

## 🎯 Objectif du projet
Ce projet vise à proposer automatiquement une **intervention de renforcement musculaire personnalisée** sur 6 semaines, à partir de mesures de force initiales et de caractéristiques individuelles (âge, sexe, poids, niveau de pratique, etc.).  
L’application utilise des **modèles de Machine Learning** entraînés sur des données expérimentales pour **équilibrer la charge (%RM)** et le **volume (séries/semaine)** afin d’optimiser les gains de force sur six groupes musculaires principaux.

---

## ⚙️ Fonctionnalités principales
- Saisie des **données personnelles** (âge, sexe, poids, latéralité, niveau, 1RM)  
- Entrée des **niveaux de force initiaux** (6 groupes musculaires)  
- Calcul automatique :
  - du **% de RM optimal**  
  - du **nombre de séries/semaine recommandé**  
  - des **gains de force estimés** pour chaque muscle  
- Interface ergonomique et esthétique réalisée avec **Streamlit**  
- Données totalement **anonymisées** pour le respect de la confidentialité  

---

## 🧩 Données utilisées
Les données proviennent d’un échantillon de **sujets sains**, comprenant :
- Informations générales (`information initiales.csv`)
- Mesures de force à J0 et à J+6 semaines
- Interventions (%RM, séries/semaine)

Toutes les données publiées dans ce dépôt sont **anonymisées** (`data/anonymes/`).

---

## 🧮 Technologies et bibliothèques
- **Langage** : Python 3.13  
- **Framework web** : Streamlit  
- **Modélisation** : scikit-learn  
- **Manipulation de données** : pandas, numpy  
- **Visualisation** : Altair (intégré à Streamlit)

---

## 💻 Utilisation locale

1️⃣ **Cloner le dépôt :**
```
git clone https://github.com/elias650/projet-reco.git
cd projet-reco
```

2️⃣ **Créer un environnement virtuel :**
```
python -m venv .venv
.venv\Scripts\activate   # sur Windows
source .venv/bin/activate  # sur macOS / Linux
```

3️⃣ **Installer les dépendances :**
```
pip install -r requirements.txt
```

4️⃣ **Lancer l’application :**
```
streamlit run app.py
```

L’application s’ouvre automatiquement dans votre navigateur à l’adresse :  
👉 [http://localhost:8501](http://localhost:8501)

---

## ☁️ Déploiement en ligne
L’application est hébergée sur **Streamlit Cloud** :  
🔗 [https://projet-reco.streamlit.app](https://projet-reco.streamlit.app)

---

## 🧑‍🔬 Auteur
Projet réalisé par **Élias Simon**,  
dans le cadre d’un **mémoire de fin d’études en kinésithérapie**,  
portant sur l’utilisation du **Machine Learning dans la prescription de rééducation personnalisée**.

---

## 📁 Organisation du dépôt
```
projet-reco/
│
├── app.py                  # Application Streamlit principale
├── modele.py               # Modèles ML et fonctions de recommandation
├── anonymiser_csv.py       # Script d’anonymisation des données
├── requirements.txt        # Dépendances Python
├── .streamlit/config.toml  # Thème visuel Streamlit
├── data/
│   └── anonymes/           # Données d’entraînement anonymisées
│
└── README.md               # (ce fichier)
```

---

## 📘 Licence
Projet académique – Usage scientifique et pédagogique uniquement.  
Toute réutilisation des données ou du code doit citer l’auteur original.
