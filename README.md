# Projet de Prévision de la Consommation Énergétique



## Table des matières
- [Présentation](#présentation)
- [Installation](#installation)
- [Description du Dataset](#description-du-dataset)
- [Étapes du Notebook](#étapes-du-notebook)
- [Détails de l'Application Flask (app.py)](#détails-de-lapplication-flask-apppy)
- [Interface Utilisateur](#interface-utilisateur)

---

## Présentation

Ce projet propose une solution de prévision de la consommation énergétique à l’aide de modèles de machine learning et peut être testé au travers d'une interface utilisateur en local afin de prédire la consommation annuelle et explorer la sensibilité de la consommation énergétique à la température réglée dans son logement.

## Installation

Pour que l'application fonctionne correctement, suivez ces étapes pour installer l'environnement et les dépendances :

1. **Cloner le dépôt GitHub** : Clonez le dépôt sur votre machine locale.
   ```bash
   git clone https://github.com/votreutilisateur/votreprojet.git
   cd Energyconsumption
   ```

2. **Création d'un environnement virtuel** :
   ```bash
   python -m venv venv
   ```

3. **Activation de l’environnement virtuel** :
   - Si vous êtes sous Windows :
     ```bash
     .\venv\Scripts\activate
     ```
   - Si vous êtes sous macOS/Linux :
     ```bash
     source venv/bin/activate
     ```

4. **Installation des dépendances** :
   ```bash
   pip install -r requirements.txt
   ```

## Description du Dataset

Le dataset, issu de Kaggle (https://www.kaggle.com/datasets/mrsimple07/energy-consumption-prediction), est `Energy_consumption.csv`. 
Il contient les données de consommation énergétique avec les colonnes suivantes :
- **Timestamp** : Horodatage de chaque relevé.
- **Temperature** : Température du bâtiment.
- **Humidity** : Taux d'humidité.
- **SquareFootage** : Surface en pieds carrés.
- **Occupancy** : Nombre de personnes dans le bâtiment.
- **HVACUsage** et **LightingUsage** : Indiquent si les systèmes de chauffage/climatisation et d'éclairage sont en marche.
- **RenewableEnergy** : Niveau d’énergie renouvelable utilisée.
- **DayOfWeek** et **Holiday** : Jour de la semaine et statut jour férié.
- **EnergyConsumption** : Consommation d'énergie en kWh.

## Étapes du Notebook

Le notebook **Energy consumption.ipynb** a pour but de séléctionner le meilleur model avec les meilleurs paramètres pour le futur focrecast.Il suit un processus en plusieurs étapes pour développer le modèle :

1. **Exploration des données** : Analyse des statistiques descriptives et visualisation des distributions pour comprendre le comportement des variables.
2. **Préparation des données** : Nettoyage des données, gestion des valeurs manquantes et transformation des variables catégorielles en indicateurs.
3. **Sélection et évaluation des modèles** : Utilisation de plusieurs modèles de régression (RandomForest, XGBoost et HistGradientBoosting) pour prédire la consommation énergétique.
4. **Validation croisée et GridSearch** : Optimisation des hyperparamètres pour chaque modèle à l’aide de GridSearchCV et stockage des meilleurs modèles dans MLflow.
5. **Entraînement final** : Entraînement du modèle choisi avec les meilleurs paramètres trouvés et sauvegarde pour utilisation dans l’application Flask.

## Détails de l'Application Flask (app.py)

Le fichier `app.py` contient le code Flask qui sert d’API et fournit une interface utilisateur pour entrer les valeurs des paramètres nécessaires à la prévision de la consommation énergétique. Les fonctionnalités principales sont :

- **Route `/`** : Accueille l’utilisateur avec un formulaire de saisie pour la prévision de la consommation énergétique.
- **Route `/forecast`** : Calcule la consommation énergétique annuelle en fonction des valeurs saisies et retourne les prévisions de consommation pour des températures normales et une température basse constante.

## Interface Utilisateur

### Page HTML - `forecast.html`

1. Page de formulaire : Saisir les paramètres environnementaux (température, humidité, etc.).
2. Page de resultat : Recevoir les résultats de prévisions comparatives de la consommation énergétique annuelle et les économies potentielles d'énergie en réduisant la température du chauffage à 0.

