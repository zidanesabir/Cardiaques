import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image
import os
from pathlib import Path
from datetime import datetime
import time

# Get the current directory
current_dir = Path(__file__).parent

# Création du dossier assets s'il n'existe pas
assets_dir = current_dir / 'assets'
if not os.path.exists(assets_dir):
    os.makedirs(assets_dir)

# Page configuration
st.set_page_config(
    page_title="Prédiction de Maladies Cardiaques",
    page_icon="❤️",
    layout="wide"
)

# Custom CSS amélioré
st.markdown("""
<style>
main {
    padding: 2rem;
    font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
}
.stButton > button {
    background-color: #ff4b4b;
    color: white;
    font-weight: bold;
    padding: 0.75rem 2.5rem;
    transition: all 0.3s ease;
    border-radius: 8px;
    border: none;
    box-shadow: 0 2px 4px rgba(0,0,0,0.1);
}
.stButton > button:hover {
    background-color: #ff6b6b;
    transform: translateY(-2px);
    box-shadow: 0 4px 8px rgba(0,0,0,0.2);
}
.stDataFrame {
    background-color: white;
    border-radius: 10px;
    box-shadow: 0 2px 8px rgba(0,0,0,0.1);
    padding: 1rem;
}
.patient-info {
    background-color: #f8f9fa;
    padding: 1.5rem;
    border-radius: 12px;
    margin-bottom: 1.5rem;
    box-shadow: 0 2px 6px rgba(0,0,0,0.05);
    border: 1px solid #e9ecef;
}
.risk-gauge {
    margin: 2.5rem 0;
    padding: 1rem;
    background-color: white;
    border-radius: 10px;
    box-shadow: 0 2px 6px rgba(0,0,0,0.05);
}
.header-container {
    display: flex;
    align-items: center;
    margin-bottom: 2rem;
    padding: 1rem;
    background-color: white;
    border-radius: 10px;
    box-shadow: 0 2px 6px rgba(0,0,0,0.05);
}
.stMarkdown {
    font-size: 1.1rem;
    line-height: 1.6;
}
h1, h2, h3 {
    color: #2c3e50;
    margin-bottom: 1rem;
}
.sidebar .sidebar-content {
    background-color: #f8f9fa;
    padding: 1.5rem;
}
</style>
""", unsafe_allow_html=True)

# Header avec titre simple
st.markdown("""
    <div class="header-container">
        <h1 class="app-title"><span style="font-family: sans-serif;">&#x2764;&#xFE0F;</span> Prédiction de Maladies Cardiaques</h1>
    </div>
""", unsafe_allow_html=True)

# Introduction
st.write('''
## Bienvenue dans l'application de prédiction de maladies cardiaques

Cette application utilise un modèle d'apprentissage automatique pour prédire 
si un patient est susceptible d'avoir une maladie cardiaque en fonction de ses données médicales.
Remplissez les informations du patient dans le panneau latéral et cliquez sur "Prédire" pour obtenir une évaluation du risque.
''')

# Séparateur
st.markdown("<hr>", unsafe_allow_html=True)

# Chargement du modèle et du scaler
@st.cache_resource
def load_model():
    try:
        model = joblib.load(current_dir / 'heart_disease_model.pkl')
        scaler = joblib.load(current_dir / 'scaler.pkl')
        return model, scaler
    except Exception as e:
        st.error(f"Erreur lors du chargement du modèle: {str(e)}")
        st.info("Vérifiez que les fichiers 'heart_disease_model.pkl' et 'scaler.pkl' sont présents dans le même dossier que l'application.")
        return None, None

try:
    model, scaler = load_model()
    model_loaded = model is not None and scaler is not None
except Exception as e:
    st.error(f"Erreur: {str(e)}")
    model_loaded = False

# Fonction de prédiction améliorée avec gestion d'erreurs
def predict_heart_disease(features):
    if not model_loaded:
        st.error("Le modèle n'a pas été chargé correctement. Impossible de faire une prédiction.")
        return None, None
        
    try:
        feature_names = ['age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg', 
                        'thalach', 'exang', 'oldpeak', 'slope', 'ca', 'thal']
        features_df = pd.DataFrame([features], columns=feature_names)
        features_scaled = scaler.transform(features_df)
        prediction = model.predict(features_scaled)
        probability = model.predict_proba(features_scaled)[0][1]
        return prediction[0], probability
    except Exception as e:
        st.error(f"Erreur lors de la prédiction: {str(e)}")
        st.info("Vérifiez que les données saisies sont valides.")
        return None, None

# Sidebar avec icône
st.sidebar.markdown("# 👨‍⚕️ Données du patient")
st.sidebar.markdown("Veuillez remplir les informations suivantes:")

# Collecte des données du patient avec explications
with st.sidebar.expander("ℹ️ Guide des paramètres", expanded=False):
    st.markdown("""
    - **Âge**: Âge du patient en années
    - **Sexe**: Genre du patient
    - **Type de douleur thoracique**: Le type de douleur ressentie par le patient
    - **Tension artérielle**: Pression artérielle en mm Hg
    - **Cholestérol**: Taux de cholestérol dans le sang en mg/dl
    - **Glycémie à jeun**: Si la glycémie à jeun est > 120 mg/dl
    - **ECG au repos**: Résultats électrocardiographiques au repos
    - **Fréquence cardiaque max**: Fréquence cardiaque maximale atteinte
    - **Angine induite par exercice**: Si l'exercice induit une angine de poitrine
    - **Dépression ST**: Dépression du segment ST induite par l'exercice
    - **Pente segment ST**: Pente du segment ST pendant l'exercice
    - **Nombre de vaisseaux**: Nombre de vaisseaux principaux colorés par fluoroscopie
    - **Thalassémie**: Type de thalassémie
    """)

# Collecte des données du patient
age = st.sidebar.slider('🔢 Âge', 18, 100, 50, help="Âge du patient en années")
sex = st.sidebar.selectbox('👥 Sexe', ['Homme', 'Femme'], help="Genre du patient")
sex = 1 if sex == 'Homme' else 0

cp = st.sidebar.selectbox('💔 Type de douleur thoracique', 
                        ['Angine typique', 'Angine atypique', 'Douleur non-angineuse', 'Asymptomatique'],
                        help="Le type de douleur thoracique ressenti par le patient")
cp_dict = {'Angine typique': 0, 'Angine atypique': 1, 'Douleur non-angineuse': 2, 'Asymptomatique': 3}
cp = cp_dict[cp]

trestbps = st.sidebar.slider('🩺 Tension artérielle au repos (mm Hg)', 80, 220, 120, 
                           help="Pression artérielle au repos en mm Hg")
chol = st.sidebar.slider('🔬 Cholestérol sérique (mg/dl)', 100, 600, 200, 
                       help="Cholestérol sérique en mg/dl")

fbs = st.sidebar.selectbox('🩸 Glycémie à jeun > 120 mg/dl', ['Non', 'Oui'], 
                         help="Si la glycémie à jeun est supérieure à 120 mg/dl")
fbs = 1 if fbs == 'Oui' else 0

restecg = st.sidebar.selectbox('📈 Résultats ECG au repos', 
                             ['Normal', 'Anomalie de l\'onde ST-T', 'Hypertrophie ventriculaire gauche'],
                             help="Résultats électrocardiographiques au repos")
restecg_dict = {'Normal': 0, 'Anomalie de l\'onde ST-T': 1, 'Hypertrophie ventriculaire gauche': 2}
restecg = restecg_dict[restecg]

thalach = st.sidebar.slider('💓 Fréquence cardiaque maximale', 70, 220, 150, 
                          help="Fréquence cardiaque maximale atteinte")

exang = st.sidebar.selectbox('🏃 Angine induite par l\'exercice', ['Non', 'Oui'], 
                           help="Si l'exercice induit une angine")
exang = 1 if exang == 'Oui' else 0

oldpeak = st.sidebar.slider('📉 Dépression ST induite par l\'exercice', 0.0, 6.0, 1.0, 0.1, 
                          help="Dépression du segment ST induite par l'exercice par rapport au repos")

slope = st.sidebar.selectbox('📊 Pente du segment ST de pointe', 
                           ['Ascendante', 'Plate', 'Descendante'],
                           help="Pente du segment ST pendant l'exercice")
slope_dict = {'Ascendante': 0, 'Plate': 1, 'Descendante': 2}
slope = slope_dict[slope]

ca_options = {
    '0 vaisseau': 0,
    '1 vaisseau': 1,
    '2 vaisseaux': 2,
    '3 vaisseaux': 3
}
ca = st.sidebar.selectbox('🫀 Nombre de vaisseaux colorés', 
                         list(ca_options.keys()),
                         help="Nombre de vaisseaux principaux colorés par fluoroscopie (0-3)")
ca = ca_options[ca]

thal = st.sidebar.selectbox('🔍 Thalassémie', ['Normal', 'Défaut fixe', 'Défaut réversible'], 
                          help="Type de thalassémie")
thal_dict = {'Normal': 3, 'Défaut fixe': 6, 'Défaut réversible': 7}
thal = thal_dict[thal]

# Création du vecteur de caractéristiques
features = [age, sex, cp, trestbps, chol, fbs, restecg, thalach, exang, oldpeak, slope, ca, thal]

# Affichage des informations du patient avec style
st.header('📋 Informations du patient')
st.markdown('<div class="patient-info">', unsafe_allow_html=True)

# Affichage des données en colonnes pour une meilleure présentation
col1, col2 = st.columns(2)

with col1:
    st.subheader("Données démographiques")
    st.markdown(f"**Âge:** {age} ans")
    st.markdown(f"**Sexe:** {'Homme' if sex == 1 else 'Femme'}")
    st.markdown(f"**Type de douleur thoracique:** {list(cp_dict.keys())[list(cp_dict.values()).index(cp)]}")
    st.markdown(f"**Tension artérielle:** {trestbps} mm Hg")
    st.markdown(f"**Cholestérol:** {chol} mg/dl")
    st.markdown(f"**Glycémie à jeun > 120 mg/dl:** {'Oui' if fbs == 1 else 'Non'}")

with col2:
    st.subheader("Données cardiaques")
    st.markdown(f"**ECG au repos:** {list(restecg_dict.keys())[list(restecg_dict.values()).index(restecg)]}")
    st.markdown(f"**Fréquence cardiaque max:** {thalach}")
    st.markdown(f"**Angine induite par exercice:** {'Oui' if exang == 1 else 'Non'}")
    st.markdown(f"**Dépression ST:** {oldpeak}")
    st.markdown(f"**Pente segment ST:** {list(slope_dict.keys())[list(slope_dict.values()).index(slope)]}")
    st.markdown(f"**Nombre de vaisseaux colorés:** {ca}")
    st.markdown(f"**Thalassémie:** {list(thal_dict.keys())[list(thal_dict.values()).index(thal)]}")

st.markdown('</div>', unsafe_allow_html=True)

# Tableau complet des données (collapsible)
with st.expander("Voir toutes les données en tableau", expanded=False):
    patient_data = pd.DataFrame({
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
    })
    st.dataframe(patient_data, hide_index=True)

# Séparateur visuel
st.markdown("<hr>", unsafe_allow_html=True)

# Prédiction avec animation et amélioration visuelle
col_button, _ = st.columns([1, 3])
with col_button:
    predict_button = st.button('🔍 le risque', use_container_width=True)

if predict_button:
    if not model_loaded:
        st.error("Impossible de faire une prédiction car le modèle n'a pas été chargé correctement.")
    else:
        with st.spinner('Analyse en cours des données médicales...'):
            # Simulation d'un délai pour l'animation
            time.sleep(0.5)
            
            prediction, probability = predict_heart_disease(features)
        
        if prediction is not None:
            st.header('🎯 Résultat de la prédiction')
            
            # Création d'une mise en page en colonnes pour le résultat
            result_col1, result_col2 = st.columns([2, 1])
            
            with result_col1:
                if prediction == 1:
                    st.error(f"⚠️ Risque élevé de maladie cardiaque")
                    st.markdown(f"### Probabilité: {probability:.2%}")
                    st.markdown("""
                    Un risque élevé suggère qu'une consultation médicale est fortement recommandée
                    pour une évaluation approfondie de la santé cardiaque du patient.
                    """)
                else:
                    st.success(f"✅ Faible risque de maladie cardiaque")
                    st.markdown(f"### Probabilité: {probability:.2%}")
                    st.markdown("""
                    Un faible risque suggère une bonne santé cardiaque selon les paramètres fournis. 
                    Des contrôles réguliers restent recommandés pour maintenir une bonne santé cardiovasculaire.
                    """)
            
            with result_col2:
                if prediction == 1:
                    st.markdown("# ⚠️")
                else:
                    st.markdown("# ✅")
            
            # Jauge de risque améliorée
            st.markdown('<div class="risk-gauge">', unsafe_allow_html=True)
            fig, ax = plt.subplots(figsize=(10, 2))
            ax.barh([0], [100], color='lightgray', height=0.5)
            
            # Couleur gradient basée sur la probabilité
            if probability <= 0.25:
                bar_color = 'green'
            elif probability <= 0.5:
                bar_color = 'yellowgreen'
            elif probability <= 0.75:
                bar_color = 'orange'
            else:
                bar_color = 'red'
                
            ax.barh([0], [probability * 100], color=bar_color, height=0.5)
            ax.set_xlim(0, 100)
            ax.set_ylim(-0.5, 0.5)
            ax.set_xlabel('Probabilité de maladie cardiaque (%)', fontweight='bold')
            ax.set_yticks([])
            
            # Ajouter des marqueurs de risque
            for i, label in zip([0, 25, 50, 75, 100], ['Très faible', 'Faible', 'Moyen', 'Élevé', 'Très élevé']):
                ax.axvline(i, color='white', linestyle='-', alpha=0.3)
                ax.text(i, -0.25, f'{i}%\n{label}', ha='center', va='center', fontsize=8)
            
            # Ligne de seuil à 50%
            ax.axvline(50, color='black', linestyle='--', alpha=0.5)
            
            plt.tight_layout()
            st.pyplot(fig)
            st.markdown('</div>', unsafe_allow_html=True)
            
            # Affichage des facteurs de risque si risque élevé
            if prediction == 1:
                st.subheader("Facteurs de risque potentiels:")
                risk_factors = []
                
                if age > 60:
                    risk_factors.append("Âge élevé")
                if sex == 1:
                    risk_factors.append("Sexe masculin")
                if cp == 0:
                    risk_factors.append("Type de douleur thoracique: Angine typique")
                if trestbps > 140:
                    risk_factors.append("Hypertension artérielle")
                if chol > 240:
                    risk_factors.append("Cholestérol élevé")
                if fbs == 1:
                    risk_factors.append("Glycémie à jeun élevée")
                if exang == 1:
                    risk_factors.append("Angine induite par exercice")
                if oldpeak > 2:
                    risk_factors.append("Dépression ST significative")
                if thalach < 120:
                    risk_factors.append("Fréquence cardiaque maximale basse")
                if ca > 0:
                    risk_factors.append(f"Présence de {ca} vaisseau(x) coloré(s)")
                
                if risk_factors:
                    for factor in risk_factors:
                        st.markdown(f"- {factor}")
                else:
                    st.markdown("- Aucun facteur de risque spécifique identifié dans les paramètres standards")

# Note importante avec style amélioré
st.markdown("""
<div style='background-color: #f8f9fa; padding: 1.5rem; border-radius: 10px; margin-top: 2rem; border-left: 5px solid #ff4b4b;'>
    <h3>Note importante</h3>
    <p>Cette application est conçue comme un outil éducatif et d'aide à la décision.
    Les prédictions sont basées sur un modèle statistique et ne constituent pas un diagnostic médical.</p>
    
    Consultez toujours un professionnel de santé qualifié pour une évaluation médicale complète.
</div>
""", unsafe_allow_html=True)

# Section expliquant les facteurs de risque généraux
with st.expander("💡 En savoir plus sur les facteurs de risque des maladies cardiaques", expanded=False):
    st.markdown("""
    ### Facteurs de risque courants des maladies cardiaques
    
    Les maladies cardiaques sont influencées par divers facteurs, notamment:
    
    - **Âge**: Le risque augmente avec l'âge
    - **Sexe**: Les hommes présentent généralement un risque plus élevé
    - **Hypertension artérielle**: Une pression sanguine élevée endommage les artères
    - **Cholestérol élevé**: Contribue à la formation de plaques dans les artères
    - **Diabète**: Augmente significativement le risque de maladie cardiaque
    - **Tabagisme**: Endommage les vaisseaux sanguins et réduit l'oxygène dans le sang
    - **Obésité**: Augmente la pression sur le cœur et les vaisseaux
    - **Sédentarité**: Un manque d'exercice contribue à d'autres facteurs de risque
    - **Antécédents familiaux**: La génétique joue un rôle important
    - **Stress**: Le stress chronique peut affecter la santé cardiaque
    
    La présence de plusieurs facteurs de risque augmente considérablement la probabilité de développer une maladie cardiaque.
    """)

# Footer avec informations et date
st.sidebar.markdown("---")
st.sidebar.info('''
**ℹ️ À propos de cette application**  
Cette application utilise un modèle d'apprentissage automatique entraîné sur 
le dataset UCI Heart Disease pour prédire les risques de maladie cardiaque.

Le modèle analyse 13 paramètres cliniques pour estimer la probabilité 
qu'un patient développe une maladie cardiaque.
''')

# Affichage de l'auteur avec la date actuelle
current_date = datetime.now().strftime("%d/%m/%Y")
st.sidebar.markdown(f"**Réalisé par Zidane Sabir**  \n🗓️ Date : {current_date}")

# Footer principal
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #777;">
    <p>❤️ Prendre soin de votre cœur aujourd'hui garantit un avenir plus sain</p>
</div>
""", unsafe_allow_html=True)