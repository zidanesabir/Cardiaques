# Projet de Prédiction de Maladies Cardiaques
# ========================================

# Table des matières
# -----------------
# 1. Chargement et prétraitement des données
# 2. Analyse exploratoire des données (EDA)
# 3. Modélisation et évaluation
# 4. Optimisation des modèles
# 5. Déploiement (création d'une interface simple avec Streamlit)

# Installation des bibliothèques nécessaires
# ------------------------------------------
# !pip install pandas numpy matplotlib seaborn scikit-learn xgboost joblib streamlit

# 1. Chargement et prétraitement des données
# -----------------------------------------
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

# Configuration pour de meilleurs graphiques
plt.style.use('fivethirtyeight')
sns.set_palette("deep")
plt.rcParams['figure.figsize'] = [12, 8]
plt.rcParams['figure.dpi'] = 100
plt.rcParams['font.size'] = 12

# Chargement des données
print("Chargement et exploration des données...")
# URL du dataset UCI Heart Disease (Cleveland)
url = "https://archive.ics.uci.edu/ml/machine-learning-databases/heart-disease/processed.cleveland.data"
column_names = [
    'age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg', 
    'thalach', 'exang', 'oldpeak', 'slope', 'ca', 'thal', 'target'
]
df = pd.read_csv(url, names=column_names, na_values='?')

# Affichage des informations sur le DataFrame
print("\nAperçu des données:")
print(df.head())
print("\nInformations sur le dataset:")
print(df.info())
print("\nStatistiques descriptives:")
print(df.describe())

# Vérification des valeurs manquantes
print("\nValeurs manquantes par colonne:")
print(df.isnull().sum())

# Gestion des valeurs manquantes
if df.isnull().sum().sum() > 0:
    # Remplacer les valeurs manquantes par la médiane pour les variables numériques
    for col in df.select_dtypes(include=['float64', 'int64']).columns:
        if df[col].isnull().sum() > 0:
            median_value = df[col].median()
            df[col].fillna(median_value, inplace=True)
    print("\nValeurs manquantes après traitement:")
    print(df.isnull().sum())

# Convertir en entiers pour les colonnes qui devraient être catégorielles
for col in ['sex', 'cp', 'fbs', 'restecg', 'exang', 'slope', 'ca', 'thal']:
    df[col] = df[col].astype(int)

# Convertir la variable cible en binaire (0: pas de maladie, 1: maladie)
df['target'] = df['target'].apply(lambda x: 1 if x > 0 else 0)

# Description des variables
variable_descriptions = {
    'age': "Âge en années",
    'sex': "Sexe (1 = homme, 0 = femme)",
    'cp': "Type de douleur thoracique (0-3)",
    'trestbps': "Tension artérielle au repos (mm Hg)",
    'chol': "Cholestérol sérique (mg/dl)",
    'fbs': "Glycémie à jeun > 120 mg/dl (1 = vrai, 0 = faux)",
    'restecg': "Résultats ECG au repos (0-2)",
    'thalach': "Fréquence cardiaque maximale atteinte",
    'exang': "Angine induite par l'exercice (1 = oui, 0 = non)",
    'oldpeak': "Dépression ST induite par l'exercice",
    'slope': "Pente du segment ST de pointe (0-2)",
    'ca': "Nombre de vaisseaux colorés par fluoroscopie (0-3)",
    'thal': "Thalassémie (3 = normal, 6 = défaut fixe, 7 = défaut réversible)",
    'target': "Diagnostic de maladie cardiaque (1 = maladie, 0 = pas de maladie)"
}

print("\nDescription des variables:")
for var, desc in variable_descriptions.items():
    print(f"{var}: {desc}")

# 2. Analyse Exploratoire des Données (EDA)
# ---------------------------------------
print("\n\nAnalyse Exploratoire des Données...")

# Distribution de la variable cible
plt.figure(figsize=(10, 6))
target_counts = df['target'].value_counts()
sns.countplot(x='target', data=df, palette=['#66b3ff', '#ff9999'])
plt.title('Distribution de la variable cible (Maladie cardiaque)')
plt.xlabel('Maladie cardiaque (0 = Non, 1 = Oui)')
plt.ylabel('Nombre de patients')
for i, count in enumerate(target_counts):
    plt.text(i, count + 10, f'{count} ({count/len(df):.1%})', 
             ha='center', va='bottom', fontsize=12)
plt.savefig('target_distribution.png')
print("Graphique de distribution de la cible sauvegardé.")

# Analyse par sexe et maladie cardiaque
plt.figure(figsize=(10, 6))
sns.countplot(x='sex', hue='target', data=df, palette=['#66b3ff', '#ff9999'])
plt.title('Distribution par sexe et maladie cardiaque')
plt.xlabel('Sexe (0 = Femme, 1 = Homme)')
plt.ylabel('Nombre de patients')
plt.legend(['Pas de maladie', 'Maladie cardiaque'])
plt.savefig('sex_target_distribution.png')
print("Graphique de distribution par sexe sauvegardé.")

# Distribution de l'âge par maladie cardiaque
plt.figure(figsize=(12, 6))
sns.histplot(data=df, x='age', hue='target', kde=True, bins=15, palette=['#66b3ff', '#ff9999'])
plt.title('Distribution de l\'âge par diagnostic de maladie cardiaque')
plt.xlabel('Âge')
plt.ylabel('Nombre de patients')
plt.savefig('age_distribution.png')
print("Graphique de distribution de l'âge sauvegardé.")

# Matrice de corrélation
plt.figure(figsize=(14, 10))
correlation_matrix = df.corr()
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt='.2f', linewidths=.5)
plt.title('Matrice de corrélation')
plt.savefig('correlation_matrix.png')
print("Matrice de corrélation sauvegardée.")

# Boxplots des variables numériques par diagnostic
numerical_features = ['age', 'trestbps', 'chol', 'thalach', 'oldpeak']
plt.figure(figsize=(15, 10))
for i, feature in enumerate(numerical_features):
    plt.subplot(2, 3, i+1)
    sns.boxplot(x='target', y=feature, data=df, palette=['#66b3ff', '#ff9999'])
    plt.title(f'{feature} par diagnostic')
    plt.xlabel('Maladie cardiaque (0 = Non, 1 = Oui)')
plt.tight_layout()
plt.savefig('boxplots.png')
print("Boxplots des variables numériques sauvegardés.")

# Relations entre les principales variables
plt.figure(figsize=(15, 10))
sns.pairplot(df[['age', 'trestbps', 'chol', 'thalach', 'target']], 
             hue='target', palette=['#66b3ff', '#ff9999'])
plt.savefig('pairplot.png')
print("Pairplot des principales variables sauvegardé.")

# 3. Modélisation et évaluation
# ---------------------------
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, cross_val_score, KFold
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from xgboost import XGBClassifier
from sklearn.metrics import (accuracy_score, precision_score, recall_score, 
                             f1_score, confusion_matrix, classification_report,
                             roc_curve, roc_auc_score, auc)
import joblib

print("\n\nPréparation des données pour la modélisation...")

# Séparation des variables
X = df.drop('target', axis=1)
y = df['target']

# Normalisation des données
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Division en ensembles d'entraînement et de test
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

print(f"Dimensions de X_train: {X_train.shape}")
print(f"Dimensions de X_test: {X_test.shape}")

# Fonction pour évaluer les modèles
def evaluate_model(model, X_train, X_test, y_train, y_test, model_name):
    # Entraînement du modèle
    model.fit(X_train, y_train)
    
    # Prédictions
    y_pred = model.predict(X_test)
    
    # Validation croisée
    cv = KFold(n_splits=5, shuffle=True, random_state=42)
    cv_scores = cross_val_score(model, X_scaled, y, cv=cv, scoring='accuracy')
    
    # Calcul des métriques
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    
    # ROC Curve et AUC
    if hasattr(model, "predict_proba"):
        y_proba = model.predict_proba(X_test)[:, 1]
        auc_score = roc_auc_score(y_test, y_proba)
    else:
        y_proba = y_pred
        auc_score = roc_auc_score(y_test, y_pred)
    
    # Affichage des résultats
    print(f"\n{model_name}:")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1-score: {f1:.4f}")
    print(f"AUC: {auc_score:.4f}")
    print(f"CV Accuracy: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")
    
    # Matrice de confusion
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title(f'Matrice de confusion - {model_name}')
    plt.xlabel('Prédiction')
    plt.ylabel('Réalité')
    plt.savefig(f'confusion_matrix_{model_name}.png')
    
    # Courbe ROC
    plt.figure(figsize=(8, 6))
    fpr, tpr, _ = roc_curve(y_test, y_proba)
    plt.plot(fpr, tpr, label=f'AUC = {auc_score:.4f}')
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlabel('Taux de faux positifs')
    plt.ylabel('Taux de vrais positifs')
    plt.title(f'Courbe ROC - {model_name}')
    plt.legend()
    plt.savefig(f'roc_curve_{model_name}.png')
    
    return {
        'model': model,
        'name': model_name,
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'auc': auc_score,
        'cv_mean': cv_scores.mean(),
        'cv_std': cv_scores.std()
    }

# Liste des modèles à évaluer
print("\nEntrainement et évaluation des modèles...")
models = [
    ('Logistic Regression', LogisticRegression(max_iter=1000, random_state=42)),
    ('Decision Tree', DecisionTreeClassifier(random_state=42)),
    ('Random Forest', RandomForestClassifier(random_state=42)),
    ('KNN', KNeighborsClassifier()),
    ('SVM', SVC(probability=True, random_state=42)),
    ('XGBoost', XGBClassifier(random_state=42))
]

# Évaluation de chaque modèle
results = []
for name, model in models:
    result = evaluate_model(model, X_train, X_test, y_train, y_test, name)
    results.append(result)

# Comparaison des modèles
def plot_model_comparison(results):
    model_names = [r['name'] for r in results]
    metrics = {
        'Accuracy': [r['accuracy'] for r in results],
        'Precision': [r['precision'] for r in results],
        'Recall': [r['recall'] for r in results],
        'F1-score': [r['f1'] for r in results],
        'AUC': [r['auc'] for r in results]
    }
    
    plt.figure(figsize=(15, 10))
    x = np.arange(len(model_names))
    width = 0.15
    multiplier = 0
    
    for metric, scores in metrics.items():
        offset = width * multiplier
        rects = plt.bar(x + offset, scores, width, label=metric)
        multiplier += 1
    
    plt.title('Comparaison des performances des modèles')
    plt.xlabel('Modèle')
    plt.ylabel('Score')
    plt.ylim(0, 1.1)
    plt.xticks(x + width * 2, model_names, rotation=45)
    plt.legend(loc='upper left')
    plt.tight_layout()
    plt.savefig('model_comparison.png')

plot_model_comparison(results)
print("Graphique de comparaison des modèles sauvegardé.")

# Identification du meilleur modèle
best_model = max(results, key=lambda x: x['f1'])
print(f"\nMeilleur modèle basé sur le F1-score: {best_model['name']} avec un F1-score de {best_model['f1']:.4f}")

# 4. Optimisation des modèles
# -------------------------
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.feature_selection import SelectKBest, f_classif, RFE
from sklearn.decomposition import PCA

print("\n\nOptimisation du meilleur modèle...")

# Sélectionner le meilleur modèle pour l'optimisation
best_model_name = best_model['name']

# Définir les paramètres de recherche selon le type de modèle
if best_model_name == 'Logistic Regression':
    param_grid = {
        'C': [0.01, 0.1, 1, 10, 100],
        'penalty': ['l1', 'l2', 'elasticnet', None],
        'solver': ['newton-cg', 'lbfgs', 'liblinear', 'sag', 'saga']
    }
    search_model = LogisticRegression(max_iter=1000, random_state=42)
    
elif best_model_name == 'Decision Tree':
    param_grid = {
        'max_depth': [None, 10, 20, 30, 40, 50],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4],
        'criterion': ['gini', 'entropy']
    }
    search_model = DecisionTreeClassifier(random_state=42)
    
elif best_model_name == 'Random Forest':
    param_grid = {
        'n_estimators': [50, 100, 200],
        'max_depth': [None, 10, 20, 30],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4]
    }
    search_model = RandomForestClassifier(random_state=42)
    
elif best_model_name == 'KNN':
    param_grid = {
        'n_neighbors': [3, 5, 7, 9, 11, 13, 15],
        'weights': ['uniform', 'distance'],
        'algorithm': ['auto', 'ball_tree', 'kd_tree', 'brute']
    }
    search_model = KNeighborsClassifier()
    
elif best_model_name == 'SVM':
    param_grid = {
        'C': [0.1, 1, 10, 100],
        'gamma': [0.001, 0.01, 0.1, 1],
        'kernel': ['rbf', 'linear', 'poly', 'sigmoid']
    }
    search_model = SVC(probability=True, random_state=42)
    
elif best_model_name == 'XGBoost':
    param_grid = {
        'n_estimators': [50, 100, 200],
        'max_depth': [3, 4, 5, 6],
        'learning_rate': [0.01, 0.05, 0.1, 0.2],
        'subsample': [0.7, 0.8, 0.9, 1.0],
        'colsample_bytree': [0.7, 0.8, 0.9, 1.0]
    }
    search_model = XGBClassifier(random_state=42)

# GridSearchCV pour trouver les meilleurs paramètres
grid_search = GridSearchCV(
    estimator=search_model,
    param_grid=param_grid,
    cv=5,
    scoring='f1',
    n_jobs=-1,
    verbose=1
)

grid_result = grid_search.fit(X_train, y_train)

print(f"Meilleurs paramètres pour {best_model_name}: {grid_result.best_params_}")
print(f"Meilleur score F1 avec ces paramètres: {grid_result.best_score_:.4f}")

# Évaluation du modèle optimisé
optimized_model = grid_result.best_estimator_
optimized_result = evaluate_model(optimized_model, X_train, X_test, y_train, y_test, f"{best_model_name} (Optimisé)")

print(f"\nAmélioration du F1-score: {optimized_result['f1'] - best_model['f1']:.4f}")

# Feature importance
if best_model_name in ['Random Forest', 'XGBoost', 'Decision Tree']:
    feature_importance = optimized_model.feature_importances_
    feature_names = X.columns
    
    # Tri des features par importance
    indices = np.argsort(feature_importance)[::-1]
    
    plt.figure(figsize=(12, 8))
    plt.title(f'Importance des variables - {best_model_name}')
    plt.bar(range(X.shape[1]), feature_importance[indices], align='center')
    plt.xticks(range(X.shape[1]), [feature_names[i] for i in indices], rotation=90)
    plt.tight_layout()
    plt.savefig('feature_importance.png')
    print("\nGraphique d'importance des variables sauvegardé.")
    
    print("\nImportance des variables:")
    for i in indices:
        print(f"{feature_names[i]}: {feature_importance[i]:.4f}")

# Sauvegarde du modèle optimisé
joblib.dump(optimized_model, 'heart_disease_model.pkl')
joblib.dump(scaler, 'scaler.pkl')
print("\nModèle optimisé et scaler sauvegardés.")

# 5. Déploiement simple avec Streamlit
# ----------------------------------
print("\n\nCréation d'une application Streamlit pour le déploiement...")

# Création d'un fichier app.py pour Streamlit
streamlit_code = """
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

# Chargement du modèle et du scaler
@st.cache_resource
def load_model():
    model = joblib.load('heart_disease_model.pkl')
    scaler = joblib.load('scaler.pkl')
    return model, scaler

model, scaler = load_model()

# Fonction de prédiction
def predict_heart_disease(features):
    features_scaled = scaler.transform([features])
    prediction = model.predict(features_scaled)
    probability = model.predict_proba(features_scaled)[0][1]
    return prediction[0], probability

# Interface utilisateur
st.title('Prédiction de Maladies Cardiaques')
st.write('''
Cette application utilise un modèle d'apprentissage automatique pour prédire 
si un patient est susceptible d'avoir une maladie cardiaque en fonction de ses données médicales.
''')

st.sidebar.header('Informations du patient')

# Collecte des données du patient
age = st.sidebar.slider('Âge', 20, 100, 50)
sex = st.sidebar.selectbox('Sexe', ['Homme', 'Femme'])
sex = 1 if sex == 'Homme' else 0

cp = st.sidebar.selectbox('Type de douleur thoracique', 
                           ['Angine typique', 'Angine atypique', 'Douleur non-angineuse', 'Asymptomatique'])
cp_dict = {'Angine typique': 0, 'Angine atypique': 1, 'Douleur non-angineuse': 2, 'Asymptomatique': 3}
cp = cp_dict[cp]

trestbps = st.sidebar.slider('Tension artérielle au repos (mm Hg)', 90, 200, 120)
chol = st.sidebar.slider('Cholestérol sérique (mg/dl)', 100, 600, 200)

fbs = st.sidebar.selectbox('Glycémie à jeun > 120 mg/dl', ['Non', 'Oui'])
fbs = 1 if fbs == 'Oui' else 0

restecg = st.sidebar.selectbox('Résultats ECG au repos', 
                               ['Normal', 'Anomalie de l\'onde ST-T', 'Hypertrophie ventriculaire gauche'])
restecg_dict = {'Normal': 0, 'Anomalie de l\'onde ST-T': 1, 'Hypertrophie ventriculaire gauche': 2}
restecg = restecg_dict[restecg]

thalach = st.sidebar.slider('Fréquence cardiaque maximale atteinte', 70, 220, 150)

exang = st.sidebar.selectbox('Angine induite par l\'exercice', ['Non', 'Oui'])
exang = 1 if exang == 'Oui' else 0

oldpeak = st.sidebar.slider('Dépression ST induite par l\'exercice', 0.0, 6.0, 1.0, 0.1)

slope = st.sidebar.selectbox('Pente du segment ST de pointe', 
                           ['Ascendante', 'Plate', 'Descendante'])
slope_dict = {'Ascendante': 0, 'Plate': 1, 'Descendante': 2}
slope = slope_dict[slope]

ca = st.sidebar.selectbox('Nombre de vaisseaux colorés par fluoroscopie', [0, 1, 2, 3])

thal = st.sidebar.selectbox('Thalassémie', ['Normal', 'Défaut fixe', 'Défaut réversible'])
thal_dict = {'Normal': 3, 'Défaut fixe': 6, 'Défaut réversible': 7}
thal = thal_dict[thal]

# Création du vecteur de caractéristiques
features = [age, sex, cp, trestbps, chol, fbs, restecg, thalach, exang, oldpeak, slope, ca, thal]

# Affichage des informations du patient
st.header('Informations du patient')
patient_data = {
    'Caractéristique': ['Âge', 'Sexe', 'Type de douleur thoracique', 'Tension artérielle', 
                      'Cholestérol', 'Glycémie à jeun > 120', 'ECG au repos', 
                      'Fréquence cardiaque max', 'Angine induite par exercice', 
                      'Dépression ST', 'Pente segment ST', 'Nombre de vaisseaux', 'Thalassémie'],
    'Valeur': [age, 'Homme' if sex == 1 else 'Femme', 
              list(cp_dict.keys())[list(cp_dict.values()).index(cp)],
              f"{trestbps} mm Hg", f"{chol} mg/dl", 
              'Oui' if fbs == 1 else 'Non',
              list(restecg_dict.keys())[list(restecg_dict.values()).index(restecg)],
              thalach,
              'Oui' if exang == 1 else 'Non',
              oldpeak,
              list(slope_dict.keys())[list(slope_dict.values()).index(slope)],
              ca,
              list(thal_dict.keys())[list(thal_dict.values()).index(thal)]]
}
st.table(pd.DataFrame(patient_data))

# Prédiction
if st.button('Prédire'):
    prediction, probability = predict_heart_disease(features)
    
    st.header('Résultat de la prédiction')
    if prediction == 1:
        st.error(f"⚠️ Risque élevé de maladie cardiaque (Probabilité: {probability:.2%})")
    else:
        st.success(f"✅ Faible risque de maladie cardiaque (Probabilité: {probability:.2%})")
    
    # Jauge de risque
    fig, ax = plt.subplots(figsize=(10, 2))
    ax.barh([0], [100], color='lightgray', height=0.5)
    ax.barh([0], [probability * 100], color='red' if probability > 0.5 else 'green', height=0.5)
    ax.set_xlim(0, 100)
    ax.set_ylim(-0.5, 0.5)
    ax.set_xlabel('Probabilité de maladie cardiaque (%)')
    ax.set_yticks([])
    for i in range(0, 101, 20):
        ax.axvline(i, color='white', linestyle='-', alpha=0.3)
        ax.text(i, -0.25, f'{i}%', ha='center', va='center')
    ax.axvline(50, color='black', linestyle='--', alpha=0.5)
    plt.tight_layout()
    st.pyplot(fig)
    
    st.write('''
    **Note importante**: Cette prédiction est fournie à titre indicatif seulement. 
    Consultez toujours un professionnel de santé pour un diagnostic médical.
    ''')

st.sidebar.info('''
**À propos de cette application**  
Cette application utilise un modèle d'apprentissage automatique entraîné sur 
le dataset UCI Heart Disease pour prédire les risques de maladie cardiaque.
''')
"""

with open('app.py', 'w', encoding='utf-8') as f:
    f.write(streamlit_code)
print("Application Streamlit créée dans le fichier 'app.py'.")
print("\nPour lancer l'application, exécutez la commande: 'streamlit run app.py'")

# Résumé du projet
print("\n\n" + "="*50)
print("RÉSUMÉ DU PROJET")
print("="*50)
print(f"Nombre d'observations: {len(df)}")
print(f"Distribution de la cible: {df['target'].value_counts().to_dict()}")
print(f"Meilleur modèle: {optimized_result['name']}")
print(f"F1-score: {optimized_result['f1']:.4f}")
print(f"Accuracy: {optimized_result['accuracy']:.4f}")
print(f"AUC: {optimized_result['auc']:.4f}")

print("\nFichiers générés:")
print("- heart_disease_model.pkl (modèle optimisé)")
print("- scaler.pkl (normalisateur)")
print("- app.py (application Streamlit)")
print("- Plusieurs graphiques d'analyse (.png)")

print("\nProjet de prédiction de maladies cardiaques terminé avec succès!")