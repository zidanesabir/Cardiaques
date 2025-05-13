import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image
import os
import base64

from pathlib import Path
from datetime import datetime
import time
import re
import pytesseract
from pdf2image import convert_from_bytes
import docx2txt
import mammoth
import io
import matplotlib.pyplot as plt
import time
from datetime import datetime
import plotly.graph_objects as go
# Ajoutez ces imports au début de votre fichier (dans la section des imports)
from fpdf import FPDF 

               # Pour les calculs numériques
            # Pour la génération de PDF
                     # Pour les pauses
# Get the current directory
current_dir = Path.cwd()

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
.upload-section {
    background-color: #f8f9fa;
    padding: 1.5rem;
    border-radius: 12px;
    margin-bottom: 1.5rem;
    box-shadow: 0 2px 6px rgba(0,0,0,0.05);
    border: 1px solid #e9ecef;
}
.info-box {
    background-color: #e3f2fd;
    padding: 1rem;
    border-radius: 8px;
    border-left: 4px solid #2196f3;
    margin-bottom: 1rem;
}
.success-box {
    background-color: #e8f5e9;
    padding: 1rem;
    border-radius: 8px;
    border-left: 4px solid #4caf50;
    margin-bottom: 1rem;
}
.warning-box {
    background-color: #fff8e1;
    padding: 1rem;
    border-radius: 8px;
    border-left: 4px solid #ff9800;
    margin-bottom: 1rem;
}
.tab-content {
    padding: 1.5rem;
    background-color: white;
    border-radius: 0 0 10px 10px;
    box-shadow: 0 2px 6px rgba(0,0,0,0.05);
}
</style>
""", unsafe_allow_html=True)

# Header with logo and title
st.markdown("""
    <div class="header-container">
        <h1><span style="font-family: sans-serif;">&#x2764;&#xFE0F;</span> Prédiction de Maladies Cardiaques</h1>
    </div>
""", unsafe_allow_html=True)

# Introduction
st.write('''
## Bienvenue dans l'application de prédiction de maladies cardiaques

Cette application utilise un modèle d'apprentissage automatique pour prédire 
si un patient est susceptible d'avoir une maladie cardiaque en fonction de ses données médicales.
Vous pouvez soit remplir manuellement les informations du patient dans le panneau latéral, soit scanner automatiquement
un document médical uploadé.
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

# Fonction pour extraire du texte à partir de différents types de documents
def extract_text_from_document(uploaded_file):
    file_extension = uploaded_file.name.split('.')[-1].lower()
    content = uploaded_file.read()
    
    try:
        if file_extension in ['jpg', 'jpeg', 'png', 'bmp', 'tiff']:
            img = Image.open(io.BytesIO(content))
            text = pytesseract.image_to_string(img, lang='fra+eng')
            return text
        
        elif file_extension == 'pdf':
            text = ""
            try:
                images = convert_from_bytes(content)
                for i, image in enumerate(images):
                    text += pytesseract.image_to_string(image, lang='fra+eng')
            except Exception as e:
                st.warning(f"Erreur lors de la conversion du PDF: {str(e)}")
                st.info("Tentative d'extraction directe...")
                # Implémenter une méthode alternative si nécessaire
            return text
        
        elif file_extension == 'docx':
            # Utiliser la bibliothèque docx2txt pour les documents Word
            try:
                return docx2txt.process(io.BytesIO(content))
            except:
                # Alternative avec mammoth
                result = mammoth.extract_raw_text(io.BytesIO(content))
                return result.value
        
        elif file_extension == 'txt':
            # Fichiers texte simples
            return content.decode('utf-8')
        
        else:
            return f"Format de fichier non pris en charge: {file_extension}"
            
    except Exception as e:
        return f"Erreur lors de l'extraction du texte: {str(e)}"

# Fonction pour extraire les paramètres médicaux du texte
def extract_medical_parameters(text):
    """
    Fonction améliorée pour extraire les paramètres médicaux à partir du texte scanné.
    Utilise des patterns plus flexibles et robustes avec une meilleure gestion des erreurs.
    """
    # Dictionnaire pour stocker les valeurs extraites
    extracted_values = {
        'age': None,
        'sex': None,
        'cp': None,
        'trestbps': None,
        'chol': None,
        'fbs': None,
        'restecg': None,
        'thalach': None,
        'exang': None,
        'oldpeak': None,
        'slope': None,
        'ca': None,
        'thal': None
    }
    
    # Conversion du texte en minuscules pour faciliter la détection
    text = text.lower()
    
    # Patterns d'extraction améliorés pour chaque paramètre
    # Ajout de variantes et de patterns plus flexibles
    patterns = {
        'age': [
            r'(?:age|âge)[:\s]*(\d{1,3})',
            r'(?:patient|personne)(?:[^\n\.]*?)(?:de|est|a|ayant)\s+(\d{1,3})\s+ans',
            r'(\d{1,3})\s+ans',
            r'né(?:e)? en (\d{4})'  # Pour calculer l'âge à partir de l'année de naissance
        ],
        'sex': [
            r'(?:sexe|genre)[:\s]*(\w+)',
            r'(?:patient|personne)\s+(\b(?:masculin|feminin|homme|femme|male|female|h|f|m)\b)',
            r'\b(?:monsieur|madame|mme|mr|m\.|mlle)\b'
        ],
        'cp': [
            r'(?:douleur thoracique|type de douleur|douleur)[:\s]*(\w+[\s\w]*)',
            r'(?:chest pain|cp|douleur)[:\s]*(\w+[\s\w]*)',
            r'(?:angine|angor)[:\s]*(\w+[\s\w]*)'
        ],
        'trestbps': [
            r'(?:tension artérielle|pression|ta|tension)[:\s]*(\d{2,3})[/\s]',
            r'(?:systolique|systolic)[:\s]*(\d{2,3})',
            r'(?:bps|trestbps)[:\s]*(\d{2,3})'
        ],
        'chol': [
            r'(?:cholestérol|chol)[:\s]*(\d{2,3})',
            r'(?:cholesterol total|cholestérol sérique|cholestérol total)[:\s]*(\d{2,3})',
            r'(?:ldl|hdl)[^\n]*?(\d{2,3})'
        ],
        'fbs': [
            r'(?:glycémie|glucose|glyc)[:\s]*((?:\d{2,3})|(?:normal|élevé|oui|non|yes|no|y|n|>|<))',
            r'(?:diabète|diabetes|diabete)[:\s]*((?:oui|non|yes|no|y|n|type|présent|absent))',
            r'(?:sucre|sugar)[:\s]*((?:\d{2,3})|(?:normal|élevé|oui|non|yes|no|y|n))'
        ],
        'restecg': [
            r'(?:ecg|électrocardiogramme|electrocardiogram)[:\s]*(\w+[\s\w]*)',
            r'(?:ecg|électrocardiogramme)[^\n]*?(normal|anormal|st|anomalie|hypertrophie)',
            r'(?:onde|wave)[:\s]*(normal|anormal|st|anomalie)'
        ],
        'thalach': [
            r'(?:fréquence cardiaque|rythme|fc max|pouls)[:\s]*(\d{2,3})',
            r'(?:heart rate|hr|rythme)[:\s]*(\d{2,3})',
            r'(?:max|maximale)[^\n]*?(\d{2,3})\s*bpm'
        ],
        'exang': [
            r'(?:angine|douleur exercice|douleur à l\'effort)[:\s]*(\w+)',
            r'(?:angine|douleur)[^\n]*?(durant|pendant|au cours|lors)[^\n]*?(effort|exercice)[^\n]*?(oui|non|yes|no|présent|absent)',
            r'(?:exercice|effort)[^\n]*?(douleur|angine)[^\n]*?(oui|non|yes|no|présent|absent)'
        ],
        'oldpeak': [
            r'(?:dépression st|depression st|dénivelé|dénivellation|st)[:\s]*([\d\.]+)',
            r'(?:depression|dépression)[^\n]*?([\d\.]+)\s*mm',
            r'(?:oldpeak)[:\s]*([\d\.]+)'
        ],
        'slope': [
            r'(?:pente st|pente|slope)[:\s]*(\w+)',
            r'(?:pente|slope)[^\n]*?(montante|ascendante|plate|descendante|up|down|flat)',
            r'(?:segment st)[^\n]*?(montante|ascendante|plate|descendante|up|down|flat)'
        ],
        'ca': [
            r'(?:vaisseaux colorés|artères|arteres|vessels|vaisseau)[:\s]*(\d)',
            r'(?:fluoroscopie|fluoroscopy)[^\n]*?(\d)\s*(?:vaisseau|artère|artere|vessel)',
            r'(?:ca|coloration)[:\s]*(\d)'
        ],
        'thal': [
            r'(?:thalassémie|thalassemie|thal)[:\s]*(\w+[\s\w]*)',
            r'(?:thalassémie|thalassemie|thal)[^\n]*?(normal|fixe|fixed|reversible|réversible)',
            r'(?:défaut|defaut|defect)[^\n]*?(normal|fixe|fixed|reversible|réversible)'
        ]
    }
    
    # Année actuelle pour calcul d'âge si nécessaire
    current_year = 2025  # Mettre à jour selon l'année actuelle
    
    # Extraire les valeurs avec les patterns
    for param, pattern_list in patterns.items():
        for pattern in pattern_list:
            matches = re.findall(pattern, text, re.IGNORECASE)
            if matches:
                # Debug - afficher ce qui a été trouvé
                st.session_state[f"debug_{param}"] = f"Trouvé: {matches[0].strip()} avec pattern {pattern}"
                # Prendre la première correspondance
                extracted_values[param] = matches[0].strip()
                break
    
    # Post-traitement des valeurs extraites
    
    # Âge - calculer à partir de l'année de naissance si nécessaire
    if extracted_values['age'] and extracted_values['age'].isdigit() and len(extracted_values['age']) == 4:
        try:
            birth_year = int(extracted_values['age'])
            if 1900 <= birth_year <= current_year:
                extracted_values['age'] = current_year - birth_year
        except:
            pass
    
    # Sexe - large gamme de possibilités
    if extracted_values['sex']:
        sex_value = extracted_values['sex'].lower()
        if any(term in sex_value for term in ['m', 'homme', 'masculin', 'male', 'mr', 'monsieur', 'h']):
            extracted_values['sex'] = 1
        elif any(term in sex_value for term in ['f', 'femme', 'féminin', 'female', 'mme', 'madame', 'mlle']):
            extracted_values['sex'] = 0
    
    # Type de douleur thoracique - plus flexible
    if extracted_values['cp']:
        cp_value = extracted_values['cp'].lower()
        if any(term in cp_value for term in ['typique', 'typical', 'type 0']):
            extracted_values['cp'] = 0
        elif any(term in cp_value for term in ['atypique', 'atypical', 'type 1']):
            extracted_values['cp'] = 1
        elif any(term in cp_value for term in ['non angineuse', 'non-anginal', 'non angineux', 'type 2']):
            extracted_values['cp'] = 2
        elif any(term in cp_value for term in ['asymptomatique', 'asymptomatic', 'asymptom', 'type 3']):
            extracted_values['cp'] = 3
    
    # Glycémie à jeun - traitement amélioré
    if extracted_values['fbs']:
        fbs_value = str(extracted_values['fbs']).lower()
        if any(term in fbs_value for term in ['oui', 'yes', 'élevé', 'high', '>', 'type', 'diabete', 'diabète']):
            extracted_values['fbs'] = 1
        elif any(term in fbs_value for term in ['non', 'no', 'normal', 'normal', '<', 'absence', 'absent']):
            extracted_values['fbs'] = 0
        else:
            # Si c'est une valeur numérique
            try:
                # Nettoyer la valeur pour supprimer les unités ou autres caractères
                fbs_num = float(''.join(c for c in fbs_value if c.isdigit() or c == '.'))
                if fbs_num > 120:
                    extracted_values['fbs'] = 1
                else:
                    extracted_values['fbs'] = 0
            except:
                pass
    
    # ECG au repos - amélioration
    if extracted_values['restecg']:
        restecg_value = extracted_values['restecg'].lower()
        if any(term in restecg_value for term in ['normal', 'normale', 'type 0']):
            extracted_values['restecg'] = 0
        elif any(term in restecg_value for term in ['st', 'anomalie', 'abnormality', 'st-t', 'onde t', 'type 1']):
            extracted_values['restecg'] = 1
        elif any(term in restecg_value for term in ['hypertrophie', 'hypertrophy', 'lvh', 'type 2']):
            extracted_values['restecg'] = 2
    
    # Angine induite par exercice - amélioration
    if extracted_values['exang']:
        exang_value = extracted_values['exang'].lower()
        if any(term in exang_value for term in ['oui', 'yes', 'positif', 'positive', 'présent', 'present']):
            extracted_values['exang'] = 1
        elif any(term in exang_value for term in ['non', 'no', 'négatif', 'negative', 'absent', 'absence']):
            extracted_values['exang'] = 0
    
    # Pente du segment ST - amélioration
    if extracted_values['slope']:
        slope_value = extracted_values['slope'].lower()
        if any(term in slope_value for term in ['ascendante', 'upsloping', 'montante', 'up', 'type 0']):
            extracted_values['slope'] = 0
        elif any(term in slope_value for term in ['plate', 'flat', 'plat', 'type 1']):
            extracted_values['slope'] = 1
        elif any(term in slope_value for term in ['descendante', 'downsloping', 'down', 'type 2']):
            extracted_values['slope'] = 2
    
    # Thalassémie - amélioration
    if extracted_values['thal']:
        thal_value = extracted_values['thal'].lower()
        if any(term in thal_value for term in ['normal', 'normale', '3']):
            extracted_values['thal'] = 3
        elif any(term in thal_value for term in ['fixe', 'fixed', 'défaut fixe', 'defect', '6']):
            extracted_values['thal'] = 6
        elif any(term in thal_value for term in ['réversible', 'reversible', 'défaut réversible', '7']):
            extracted_values['thal'] = 7
    
    # Convertir les valeurs numériques en nombres
    for param in ['age', 'trestbps', 'chol', 'thalach', 'oldpeak', 'ca']:
        if extracted_values[param] and isinstance(extracted_values[param], str):
            try:
                # Suppression des caractères non numériques (sauf point décimal)
                cleaned_value = ''.join(c for c in extracted_values[param] if c.isdigit() or c == '.')
                if cleaned_value:
                    extracted_values[param] = float(cleaned_value)
                    # Pour les paramètres entiers
                    if param in ['age', 'trestbps', 'chol', 'thalach', 'ca']:
                        extracted_values[param] = int(extracted_values[param])
            except:
                pass
    
    # Debug - journaliser les valeurs extraites
    st.session_state.debug_extraction = {k: str(v) for k, v in extracted_values.items()}
    
    return extracted_values

# Initialisation des variables de session
if 'extracted_values' not in st.session_state:
    st.session_state.extracted_values = None
if 'document_text' not in st.session_state:
    st.session_state.document_text = None
# Initialiser le flag pour appliquer les valeurs extraites
if 'apply_extracted_values' not in st.session_state:
    st.session_state.apply_extracted_values = False
# Initialiser le type d'entrée dans st.session_state
if 'input_method' not in st.session_state:
    st.session_state.input_method = "👆 Saisie manuelle"

# Sidebar avec icône
st.sidebar.markdown("# 👨‍⚕️ Données du patient")
input_method = st.sidebar.radio("Choisissez la méthode d'entrée des données:", 
                             ["👆 Saisie manuelle", "📄 Scanner un document"],
                             key="input_method")  # Utiliser key pour lier à st.session_state

# Section scan de document (si cette option est sélectionnée)
if input_method == "📄 Scanner un document":
    st.sidebar.markdown("## 📄 Uploader un document médical")
    
    # Added a unique key to this file_uploader
    uploaded_file = st.sidebar.file_uploader("Choisissez un fichier médical (PDF, image, DOCX, TXT)", 
                                        type=["pdf", "jpg", "jpeg", "png", "docx", "txt"],
                                        key="sidebar_uploader")
    
    if uploaded_file is not None:
        scan_button = st.sidebar.button("🔍 Scanner le document", use_container_width=True)
        
        if scan_button:
            with st.spinner('Extraction des données médicales en cours...'):
                # Extraire le texte du document
                text = extract_text_from_document(uploaded_file)
                st.session_state.document_text = text
                
                # Extraire les paramètres médicaux
                extracted_values = extract_medical_parameters(text)
                st.session_state.extracted_values = extracted_values
                
                # Activer l'utilisation automatique des valeurs extraites
                st.session_state.apply_extracted_values = True
                
            st.sidebar.success("✅ Document scanné avec succès!")
            
            # Option pour voir le texte brut extrait
            with st.sidebar.expander("Voir le texte extrait", expanded=False):
                st.text_area("Texte brut", st.session_state.document_text, height=300)

        # Si des valeurs ont été extraites, afficher un tableau récapitulatif
        if st.session_state.extracted_values:
            st.sidebar.markdown("### Paramètres détectés")
            
            # Créer un DataFrame pour l'affichage des paramètres extraits
            params_detected = []
            for param, value in st.session_state.extracted_values.items():
                if value is not None:
                    params_detected.append({"Paramètre": param, "Valeur détectée": value})
            
            if params_detected:
                st.sidebar.table(pd.DataFrame(params_detected))
            else:
                st.sidebar.warning("Aucun paramètre médical n'a pu être détecté dans le document.")

# Fonction pour obtenir la valeur par défaut en tenant compte des valeurs extraites
def get_default_value(param, default):
    if st.session_state.apply_extracted_values and st.session_state.extracted_values:
        if st.session_state.extracted_values[param] is not None:
            return st.session_state.extracted_values[param]
    return default

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
age = st.sidebar.slider('🔢 Âge', 18, 100, 
                       int(get_default_value('age', 50)), 
                       help="Âge du patient en années")

sex_options = ['Homme', 'Femme']
default_sex_index = 0
if st.session_state.apply_extracted_values and st.session_state.extracted_values:
    if st.session_state.extracted_values['sex'] is not None:
        try:
            sex_value = int(st.session_state.extracted_values['sex'])
            default_sex_index = 0 if sex_value == 1 else 1
        except (ValueError, TypeError):
            default_sex_index = 0
        
sex = st.sidebar.selectbox('👥 Sexe', sex_options, index=default_sex_index, 
                         help="Genre du patient")
sex = 1 if sex == 'Homme' else 0

cp_options = ['Angine typique', 'Angine atypique', 'Douleur non-angineuse', 'Asymptomatique']
cp_dict = {'Angine typique': 0, 'Angine atypique': 1, 'Douleur non-angineuse': 2, 'Asymptomatique': 3}
default_cp_index = 0
if st.session_state.apply_extracted_values and st.session_state.extracted_values:
    if st.session_state.extracted_values['cp'] is not None:
        try:
            cp_value = int(st.session_state.extracted_values['cp'])
            if 0 <= cp_value <= 3:  # Valider la plage
                default_cp_index = cp_value
        except (ValueError, TypeError):
            default_cp_index = 0

cp = st.sidebar.selectbox('💔 Type de douleur thoracique', cp_options, 
                        index=default_cp_index, 
                        help="Le type de douleur thoracique ressenti par le patient")
cp = cp_dict[cp]

# Pour restecg
restecg_options = ['Normal', 'Anomalie de l\'onde ST-T', 'Hypertrophie ventriculaire gauche']
restecg_dict = {'Normal': 0, 'Anomalie de l\'onde ST-T': 1, 'Hypertrophie ventriculaire gauche': 2}
default_restecg_index = 0
if st.session_state.apply_extracted_values and st.session_state.extracted_values:
    if st.session_state.extracted_values['restecg'] is not None:
        try:
            restecg_value = int(st.session_state.extracted_values['restecg'])
            if 0 <= restecg_value <= 2:  # Valider la plage
                default_restecg_index = restecg_value
        except (ValueError, TypeError):
            default_restecg_index = 0
            
restecg = st.sidebar.selectbox('📈 Résultats ECG au repos', restecg_options,
                             index=default_restecg_index,
                             help="Résultats électrocardiographiques au repos")
restecg = restecg_dict[restecg]

# Pour slope
slope_options = ['Ascendante', 'Plate', 'Descendante']
slope_dict = {'Ascendante': 0, 'Plate': 1, 'Descendante': 2}
default_slope_index = 0
if st.session_state.apply_extracted_values and st.session_state.extracted_values:
    if st.session_state.extracted_values['slope'] is not None:
        try:
            slope_value = int(st.session_state.extracted_values['slope'])
            if 0 <= slope_value <= 2:  # Valider la plage
                default_slope_index = slope_value
        except (ValueError, TypeError):
            default_slope_index = 0
            
slope = st.sidebar.selectbox('📊 Pente du segment ST de pointe', slope_options,
                           index=default_slope_index,
                           help="Pente du segment ST pendant l'exercice")
slope = slope_dict[slope]

# Pour ca
ca_options = {
    '0 vaisseau': 0,
    '1 vaisseau': 1,
    '2 vaisseaux': 2,
    '3 vaisseaux': 3
}
default_ca_index = 0
if st.session_state.apply_extracted_values and st.session_state.extracted_values:
    if st.session_state.extracted_values['ca'] is not None:
        try:
            ca_value = int(st.session_state.extracted_values['ca'])
            if 0 <= ca_value <= 3:
                default_ca_index = ca_value
        except (ValueError, TypeError):
            default_ca_index = 0
            
ca = st.sidebar.selectbox('🫀 Nombre de vaisseaux colorés', 
                         list(ca_options.keys()),
                         index=default_ca_index, 
                         help="Nombre de vaisseaux principaux colorés par fluoroscopie (0-3)")
ca = ca_options[ca]

# Define fasting blood sugar (fbs)
fbs_options = ['Non', 'Oui']
default_fbs_index = 0
if st.session_state.apply_extracted_values and st.session_state.extracted_values:
    if st.session_state.extracted_values['fbs'] is not None:
        try:
            fbs_value = int(st.session_state.extracted_values['fbs'])
            if fbs_value in [0, 1]:
                default_fbs_index = fbs_value
        except (ValueError, TypeError):
            default_fbs_index = 0
            
fbs = st.sidebar.selectbox('🍬 Glycémie à jeun > 120 mg/dl', fbs_options, 
                         index=default_fbs_index, 
                         help="Si la glycémie à jeun est supérieure à 120 mg/dl")
fbs = 1 if fbs == 'Oui' else 0

# Pour thal
thal_options = ['Normal', 'Défaut fixe', 'Défaut réversible']
thal_dict = {'Normal': 3, 'Défaut fixe': 6, 'Défaut réversible': 7}
default_thal_index = 0
if st.session_state.apply_extracted_values and st.session_state.extracted_values:
    if st.session_state.extracted_values['thal'] is not None:
        try:
            thal_value = int(st.session_state.extracted_values['thal'])
            if thal_value in [3, 6, 7]:
                default_thal_index = list(thal_dict.values()).index(thal_value)
        except (ValueError, TypeError):
            default_thal_index = 0
            
thal = st.sidebar.selectbox('🔍 Thalassémie', thal_options, 
                          index=default_thal_index, 
                          help="Type de thalassémie")
thal = thal_dict[thal]


trestbps = st.sidebar.slider('🩺 Tension artérielle au repos (mm Hg)', 80, 220, 
                           int(get_default_value('trestbps', 120)), 
                           help="Pression artérielle au repos en mm Hg")

# Add the chol variable definition if it's also missing
chol = st.sidebar.slider('🔬 Cholestérol sérique (mg/dl)', 100, 600, 
                       int(get_default_value('chol', 200)), 
                       help="Cholestérol sérique en mg/dl")

# Make sure the thalach variable is defined
thalach = st.sidebar.slider('💓 Fréquence cardiaque maximale', 70, 220, 
                          int(get_default_value('thalach', 150)), 
                          help="Fréquence cardiaque maximale atteinte")

# Make sure the exang variable is defined
exang_options = ['Non', 'Oui']
default_exang_index = 0
if input_method == "📄 Scanner un document" and st.session_state.extracted_values:
    if st.session_state.extracted_values['exang'] is not None:
        try:
            exang_value = int(st.session_state.extracted_values['exang'])
            if exang_value in [0, 1]:
                default_exang_index = exang_value
        except (ValueError, TypeError):
            default_exang_index = 0
            
exang = st.sidebar.selectbox('🏃 Angine induite par l\'exercice', exang_options, 
                           index=default_exang_index, 
                           help="Si l'exercice induit une angine")
exang = 1 if exang == 'Oui' else 0

# Make sure the oldpeak variable is defined
oldpeak = st.sidebar.slider('📉 Dépression ST induite par l\'exercice', 0.0, 6.0, 
                          float(get_default_value('oldpeak', 1.0)), 0.1, 
                          help="Dépression du segment ST induite par l'exercice par rapport au repos")

# Then create the features list after all variables are defined
features = [age, sex, cp, trestbps, chol, fbs, restecg, thalach, exang, oldpeak, slope, ca, thal]

# Interface principale - Afficher les onglets
# Interface principale - Afficher les onglets
tabs = st.tabs(["⚕️ Données du patient", "📄 Scan de document", "📊 Prédiction"])

# Onglet 1: Données du patient
with tabs[0]:
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
    
    # Si le document a été scanné, afficher le mode d'acquisition des données
    if input_method == "📄 Scanner un document" and st.session_state.extracted_values:
        st.success("✅ Les données ont été extraites et complétées à partir du document scanné.")
    
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

# Onglet 2: Scan de document
with tabs[1]:
    st.header('📄 Scanner un document médical')
    
    st.markdown("""
    <div class="info-box">
        <h3>🔍 Extraction automatique des données médicales</h3>
        <p>Cette fonctionnalité vous permet d'uploader un document médical (comme un rapport de laboratoire ou un dossier médical) 
        et d'extraire automatiquement les informations pertinentes pour la prédiction de maladies cardiaques.</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Zone d'upload de document
    st.markdown('<div class="upload-section">', unsafe_allow_html=True)
    uploaded_file = st.file_uploader("Choisissez un fichier médical (PDF, image, DOCX, TXT)", 
                                   type=["pdf", "jpg", "jpeg", "png", "docx", "txt"])
    
    col1, col2 = st.columns([1, 1])
    with col1:
        scan_button = st.button("🔍 Scanner et extraire les données", use_container_width=True, 
                              disabled=uploaded_file is None)
    with col2:
        clear_button = st.button("🗑️ Effacer les résultats", use_container_width=True, 
                               disabled=st.session_state.document_text is None)
    
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Traitement du document
    if scan_button and uploaded_file is not None:
        with st.spinner('Analyse du document en cours...'):
            # Simuler un délai pour l'animation
            time.sleep(1)
            
            # Extraire le texte du document
            text = extract_text_from_document(uploaded_file)
            st.session_state.document_text = text
            
            # Extraire les paramètres médicaux
            extracted_values = extract_medical_parameters(text)
            st.session_state.extracted_values = extracted_values
            
            # Vérifier le nombre de paramètres extraits
            nb_params_found = sum(1 for v in extracted_values.values() if v is not None)
            
        # Afficher les résultats
        if nb_params_found > 0:
            st.success(f"✅ Extraction réussie! {nb_params_found} paramètres médicaux ont été détectés.")
        else:
            st.warning("⚠️ Aucun paramètre médical n'a pu être détecté dans le document.")
    
    # Effacer les résultats si demandé
    if clear_button:
        st.session_state.document_text = None
        st.session_state.extracted_values = None
        st.info("Les résultats ont été effacés.")
    
    # Afficher les résultats si disponibles
    if st.session_state.document_text is not None:
        # Afficher le texte extrait dans un expander
        with st.expander("Voir le texte extrait du document", expanded=False):
            st.text_area("Texte brut", st.session_state.document_text, height=200)
        
        # Afficher les paramètres détectés
        if st.session_state.extracted_values:
            st.subheader("📋 Paramètres médicaux détectés")
            
            # Transformer les valeurs extraites en un format lisible
            readable_values = []
            for param, value in st.session_state.extracted_values.items():
                if value is not None:
                    # Convertir les codes en texte lisible
                    if param == 'sex':
                        try:
                            readable_value = 'Homme' if int(value) == 1 else 'Femme'
                        except (ValueError, TypeError):
                            readable_value = value
                    elif param == 'cp':
                        cp_labels = ['Angine typique', 'Angine atypique', 'Douleur non-angineuse', 'Asymptomatique']
                        try:
                            value_int = int(value)
                            readable_value = cp_labels[value_int] if 0 <= value_int < len(cp_labels) else value
                        except (ValueError, TypeError):
                            readable_value = value
                    elif param == 'fbs':
                        try:
                            readable_value = 'Oui' if int(value) == 1 else 'Non'
                        except (ValueError, TypeError):
                            readable_value = value
                    elif param == 'restecg':
                        restecg_labels = ['Normal', 'Anomalie de l\'onde ST-T', 'Hypertrophie ventriculaire gauche']
                        try:
                            value_int = int(value)
                            readable_value = restecg_labels[value_int] if 0 <= value_int < len(restecg_labels) else value
                        except (ValueError, TypeError):
                            readable_value = value
                    elif param == 'exang':
                        try:
                            readable_value = 'Oui' if int(value) == 1 else 'Non'
                        except (ValueError, TypeError):
                            readable_value = value
                    elif param == 'slope':
                        slope_labels = ['Ascendante', 'Plate', 'Descendante']
                        try:
                            value_int = int(value)
                            readable_value = slope_labels[value_int] if 0 <= value_int < len(slope_labels) else value
                        except (ValueError, TypeError):
                            readable_value = value
                    elif param == 'thal':
                        thal_labels = {3: 'Normal', 6: 'Défaut fixe', 7: 'Défaut réversible'}
                        try:
                            value_int = int(value)
                            readable_value = thal_labels.get(value_int, value)
                        except (ValueError, TypeError):
                            readable_value = value
                    else:
                        readable_value = value
                        
                    readable_values.append({"Paramètre": param, "Valeur détectée": readable_value})
            
            # Afficher en tableau
            if readable_values:
                st.table(pd.DataFrame(readable_values))
                
            #     # Bouton pour utiliser ces valeurs
            #     if st.button("✅ Utiliser ces valeurs pour la prédiction", use_container_width=True):
            #         # Mettre à jour la méthode d'entrée dans la sidebar
            #         st.session_state.input_method = "📄 Scanner un document"
            #         st.info("Les valeurs ont été appliquées. Allez dans l'onglet 'Prédiction' pour voir le résultat.")
            # else:
            #     st.info("Aucun paramètre pertinent n'a été détecté dans le document.")

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
        
    # Affichage de l'origine des données si elles ont été extraites d'un document
    if input_method == "📄 Scanner un document" and st.session_state.extracted_values:
        st.info("ℹ️ Certaines de ces données ont été extraites automatiquement du document scanné. Veuillez vérifier leur exactitude.")

# Affichage de l'auteur avec la date actuelle
current_date = datetime.now().strftime("%d/%m/%Y")
st.sidebar.markdown(f"**Réalisé par Zidane Sabir**  \n🗓️ Date : {current_date}")
# Onglet 3: Prédiction des maladies cardiaques
with tabs[2]:
    st.header('📊 Prédiction du risque cardiaque')
    
    # Boîte d'information avec explication détaillée
    st.markdown("""
    <div style="background-color: #f8f9fa; padding: 15px; border-radius: 10px; border-left: 5px solid #4e73df;">
        <h3>🔍 Analyse du risque de maladie cardiaque</h3>
        <p>Cette section utilise un modèle d'intelligence artificielle pour analyser vos données médicales et 
        évaluer le risque potentiel de maladie cardiaque. Le résultat fourni est une estimation statistique 
        et doit être interprété par un professionnel de santé.</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Organisation des données en colonnes pour une meilleure lisibilité
    st.subheader("📋 Récapitulatif des données")
    
    # Fonction pour récupérer les labels à partir des dictionnaires
    def get_label_from_dict(dictionary, value):
        try:
            return list(dictionary.keys())[list(dictionary.values()).index(value)]
        except (ValueError, IndexError):
            return "Non spécifié"
    
    # Organisation des données en 3 colonnes pour une meilleure répartition
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("**💼 Données démographiques**")
        st.markdown(f"• Âge: **{age}** ans")
        st.markdown(f"• Sexe: **{'Homme' if sex == 1 else 'Femme'}**")
        st.markdown(f"• Glycémie > 120: **{'Oui' if fbs == 1 else 'Non'}**")
        
    with col2:
        st.markdown("**❤️ Données cardiaques**")
        st.markdown(f"• Type de douleur: **{get_label_from_dict(cp_dict, cp)}**")
        st.markdown(f"• Tension art.: **{trestbps}** mm Hg")
        st.markdown(f"• Cholestérol: **{chol}** mg/dl")
        st.markdown(f"• Fréq. cardiaque max: **{thalach}**")
        
    with col3:
        st.markdown("**📈 Données ECG**")
        st.markdown(f"• ECG au repos: **{get_label_from_dict(restecg_dict, restecg)}**")
        st.markdown(f"• Angine d'effort: **{'Oui' if exang == 1 else 'Non'}**")
        st.markdown(f"• Dépression ST: **{oldpeak}**")
        st.markdown(f"• Pente ST: **{get_label_from_dict(slope_dict, slope)}**")
        st.markdown(f"• Vaisseaux colorés: **{ca}**")
        st.markdown(f"• Thalassémie: **{get_label_from_dict(thal_dict, thal)}**")
    
    # Séparateur visuel
    st.markdown("<hr style='margin: 30px 0; border-top: 1px solid #ddd;'>", unsafe_allow_html=True)
    
    # Bouton de prédiction amélioré
    predict_button = st.button("🩺 Analyser les données et prédire le risque", 
                              key="predict_button", 
                              use_container_width=True,
                              help="Cliquez pour lancer l'analyse avec les données saisies")
    
    if predict_button:
        if not model_loaded:
            st.error("⚠️ Le modèle de prédiction n'a pas pu être chargé. Impossible d'effectuer une analyse.")
            st.info("Vérifiez que les fichiers 'heart_disease_model.pkl' et 'scaler.pkl' sont présents et accessibles.")
        else:
            # Animation de chargement
            with st.spinner('Analyse en cours... Veuillez patienter.'):
                try:
                    # Effectuer la prédiction
                    prediction, probability = predict_heart_disease(features)
                    
                    # Courte pause pour l'animation
                    time.sleep(0.7)
                    
                    # Vérifier que la prédiction est valide
                    if prediction is None or probability is None:
                        raise ValueError("Résultat de prédiction invalide")
                        
                except Exception as e:
                    st.error(f"❌ Une erreur s'est produite lors de l'analyse: {str(e)}")
                    st.info("Veuillez vérifier les données saisies et réessayer.")
                    prediction, probability = None, None
            
            # Si la prédiction est réussie
            if prediction is not None and probability is not None:
                # Calcul des niveaux de risque et des couleurs correspondantes
                risk_percentage = probability * 100
                
                if risk_percentage >= 70:
                    risk_level = "élevé"
                    risk_color = "#FF4B4B"  # Rouge
                elif risk_percentage >= 30:
                    risk_level = "modéré"
                    risk_color = "#FFA500"  # Orange
                else:
                    risk_level = "faible"
                    risk_color = "#00CC96"  # Vert
                
                # Message de succès
                st.success("✅ Analyse terminée avec succès!")
                
                # Affichage des résultats
                st.subheader("📋 Résultats de l'analyse")
                
                # Affichage du résultat avec un design amélioré
                if prediction == 1:
                    st.markdown(f"""
                    <div style="background-color: #FFF0F0; padding: 20px; border-radius: 10px; 
                               margin-bottom: 20px; border-left: 5px solid {risk_color}; box-shadow: 0 2px 5px rgba(0,0,0,0.1);">
                        <h2 style="color: {risk_color};">⚠️ Risque de maladie cardiaque détecté</h2>
                        <p style="font-size: 18px;">Le modèle prédit un <b>risque {risk_level}</b> de maladie cardiaque 
                        avec une probabilité de <b>{risk_percentage:.1f}%</b>.</p>
                        <p style="font-style: italic; color: #555;">Une consultation médicale approfondie est recommandée.</p>
                    </div>
                    """, unsafe_allow_html=True)
                else:
                    st.markdown(f"""
                    <div style="background-color: #F0FFF0; padding: 20px; border-radius: 10px; 
                              margin-bottom: 20px; border-left: 5px solid #00CC96; box-shadow: 0 2px 5px rgba(0,0,0,0.1);">
                        <h2 style="color: #00CC96;">✅ Risque de maladie cardiaque faible</h2>
                        <p style="font-size: 18px;">Le modèle prédit un <b>risque faible</b> de maladie cardiaque 
                        avec une probabilité de <b>{100-risk_percentage:.1f}%</b> d'absence de maladie.</p>
                        <p style="font-style: italic; color: #555;">Continuez à maintenir un mode de vie sain et à effectuer des contrôles réguliers.</p>
                    </div>
                    """, unsafe_allow_html=True)
                
                # Visualisations avec onglets
                st.subheader("📊 Visualisation du risque")
                viz_tab1, viz_tab2, viz_tab3 = st.tabs(["📉 Jauge de risque", "🔄 Graphique en anneau", "📊 Comparaison"])
                
                with viz_tab1:
                    # Jauge Plotly améliorée
                    fig = go.Figure(go.Indicator(
                        mode="gauge+number+delta",
                        value=risk_percentage,
                        domain={'x': [0, 1], 'y': [0, 1]},
                        title={'text': "Probabilité de risque cardiaque (%)", 'font': {'size': 24}},
                        delta={'reference': 50, 'increasing': {'color': "#FF4B4B"}, 'decreasing': {'color': "#00CC96"}},
                        gauge={
                            'axis': {'range': [0, 100], 'tickwidth': 1, 'tickcolor': "darkblue"},
                            'bar': {'color': "darkblue"},
                            'bgcolor': "white",
                            'borderwidth': 2,
                            'bordercolor': "gray",
                            'steps': [
                                {'range': [0, 30], 'color': '#00CC96'},
                                {'range': [30, 70], 'color': '#FFA500'},
                                {'range': [70, 100], 'color': '#FF4B4B'}
                            ],
                            'threshold': {
                                'line': {'color': "red", 'width': 4},
                                'thickness': 0.75,
                                'value': risk_percentage
                            }
                        }
                    ))
                    
                    fig.update_layout(
                        height=350,
                        margin=dict(l=20, r=20, t=50, b=20),
                        paper_bgcolor="white",
                        font={'color': "darkblue", 'family': "Arial"}
                    )
                    
                    st.plotly_chart(fig, use_container_width=True)
                
                with viz_tab2:
                    # Graphique en anneau amélioré
                    fig, ax = plt.subplots(figsize=(5, 5))
                    
                    # Créer un graphique en anneau
                    wedges, texts = ax.pie(
                        [risk_percentage, 100-risk_percentage], 
                        colors=[risk_color, '#f0f0f0'], 
                        startangle=90, 
                        counterclock=False,
                        wedgeprops={'width': 0.4, 'edgecolor': 'white', 'linewidth': 2},
                        shadow=True
                    )
                    
                    # Ajouter le texte au centre
                    ax.text(0, 0, f"{risk_percentage:.1f}%", ha='center', va='center', fontsize=28, fontweight='bold')
                    ax.text(0, -0.15, "RISQUE", ha='center', va='center', fontsize=12)
                    
                    # Ajouter des annotations
                    if prediction == 1:
                        ax.annotate(f'Risque {risk_level}', xy=(0.5, 0), 
                                  xytext=(1.1, 0.5), textcoords='offset points',
                                  size=14, ha='left', va='center')
                    else:
                        ax.annotate('Risque faible', xy=(0.5, 0), 
                                  xytext=(1.1, 0.5), textcoords='offset points',
                                  size=14, ha='left', va='center')
                    
                    # Ajuster l'aspect
                    ax.set_aspect('equal')
                    plt.tight_layout()
                    
                    st.pyplot(fig)
                
                with viz_tab3:
                    # Comparaison à la population générale
                    st.markdown("### Comparaison avec la population générale")
                    
                    # Données pour la comparaison (à adapter selon les données réelles)
                    population_risk = 12  # Risque moyen dans la population générale (%)
                    
                    # Graphique en barres pour la comparaison
                    fig, ax = plt.subplots(figsize=(8, 5))
                    
                    bars = ax.bar(['Votre risque', 'Population générale'], 
                                 [risk_percentage, population_risk],
                                 color=[risk_color, '#1E90FF'])
                    
                    # Ajouter les valeurs au-dessus des barres
                    for bar in bars:
                        height = bar.get_height()
                        ax.text(bar.get_x() + bar.get_width()/2., height + 1,
                              f'{height:.1f}%', ha='center', va='bottom', fontsize=12)
                    
                    ax.set_ylabel('Probabilité de risque (%)')
                    ax.set_title('Comparaison de votre risque avec la population générale')
                    ax.set_ylim(0, max(risk_percentage, population_risk) * 1.2)
                    
                    st.pyplot(fig)
                    
                    # Message contextuel
                    if risk_percentage > population_risk:
                        st.warning(f"⚠️ Votre risque est **{risk_percentage - population_risk:.1f}%** plus élevé que la moyenne de la population générale.")
                    else:
                        st.success(f"✅ Votre risque est **{population_risk - risk_percentage:.1f}%** plus bas que la moyenne de la population générale.")
                
                # Analyse des facteurs de risque
                st.subheader("🔍 Analyse des facteurs de risque")
                
                # Définition des facteurs de risque avec poids d'importance
                risk_conditions = [
                    (age > 60, "Âge élevé", f"{age} ans", 
                     "L'âge est un facteur de risque important pour les maladies cardiovasculaires.", 3),
                    
                    (sex == 1, "Sexe masculin", "Homme", 
                     "Les hommes ont généralement un risque plus élevé de maladie cardiaque.", 2),
                    
                    (cp == 0, "Douleur thoracique typique", "Angine typique", 
                     "La douleur thoracique typique est souvent liée à des problèmes cardiaques.", 3),
                    
                    (trestbps > 140, "Hypertension", f"{trestbps} mm Hg", 
                     "Une tension artérielle élevée augmente le risque de maladie cardiaque.", 3),
                    
                    (chol > 240, "Cholestérol élevé", f"{chol} mg/dl", 
                     "Un taux de cholestérol élevé est un facteur de risque majeur.", 3),
                    
                    (fbs == 1, "Glycémie élevée", "Glycémie > 120 mg/dl", 
                     "Une glycémie élevée à jeun est un indicateur de risque cardiaque.", 2),
                    
                    (thalach < 120, "Fréquence cardiaque basse", f"{thalach} bpm", 
                     "Une fréquence cardiaque maximale faible peut indiquer un problème cardiaque.", 2),
                    
                    (exang == 1, "Angine d'effort", "Positive", 
                     "La présence d'angine à l'effort est un signe d'alerte important.", 3),
                    
                    (oldpeak > 2, "Dépression ST élevée", f"{oldpeak}", 
                     "Une dépression ST importante à l'effort est préoccupante.", 3),
                    
                    (slope == 2, "Pente ST descendante", "Descendante", 
                     "Une pente ST descendante est associée à un risque plus élevé.", 2),
                    
                    (ca > 1, "Vaisseaux colorés multiples", f"{ca} vaisseaux", 
                     "Un nombre élevé de vaisseaux colorés indique des problèmes circulatoires.", 3),
                    
                    (thal in [6, 7], "Anomalie de thalassémie", 
                     f"{get_label_from_dict(thal_dict, thal)}", 
                     "Des anomalies de thalassémie peuvent indiquer des problèmes cardiaques.", 2)
                ]
                
                # Identifier les facteurs de risque présents
                risk_factors = [(factor, value, explanation, weight) 
                               for condition, factor, value, explanation, weight in risk_conditions 
                               if condition]
                
                # Afficher les facteurs de risque identifiés
                if risk_factors:
                    # Trier par poids d'importance
                    risk_factors.sort(key=lambda x: x[3], reverse=True)
                    
                    # Créer un DataFrame sans le poids
                    risk_table = pd.DataFrame([(f, v, e) for f, v, e, _ in risk_factors], 
                                            columns=["Facteur de risque", "Valeur", "Explication"])
                    
                    st.markdown("#### Facteurs de risque identifiés")
                    st.dataframe(
                        risk_table,
                        hide_index=True,
                        use_container_width=True,
                        column_config={
                            "Facteur de risque": st.column_config.TextColumn(
                                "Facteur de risque",
                                width="medium",
                                help="Facteurs identifiés comme présentant un risque"
                            ),
                            "Valeur": st.column_config.TextColumn(
                                "Valeur",
                                width="small"
                            ),
                            "Explication": st.column_config.TextColumn(
                                "Explication",
                                width="large"
                            )
                        }
                    )
                    
                    # Visualisation des facteurs les plus importants
                    if len(risk_factors) > 1:
                        st.markdown("#### Importance relative des facteurs")
                        
                        # Limiter aux 5 facteurs les plus importants
                        top_factors = risk_factors[:5]
                        
                        fig, ax = plt.subplots(figsize=(10, 4))
                        factors = [f for f, _, _, _ in top_factors]
                        weights = [w for _, _, _, w in top_factors]
                        
                        # Dégradé de couleurs basé sur le niveau de risque
                        colors = plt.cm.Reds(np.linspace(0.5, 0.9, len(factors)))
                        
                        bars = ax.barh(factors, weights, color=colors)
                        ax.set_xlabel('Importance relative')
                        ax.set_title('Facteurs de risque les plus significatifs')
                        
                        # Valeurs à la fin de chaque barre
                        for bar in bars:
                            width = bar.get_width()
                            ax.text(width + 0.1, bar.get_y() + bar.get_height()/2, 
                                  f'{width}', ha='left', va='center')
                        
                        plt.tight_layout()
                        st.pyplot(fig)
                else:
                    st.info("✅ Aucun facteur de risque majeur n'a été identifié dans les données fournies.")
                
                # Recommandations personnalisées
                st.subheader("🩺 Recommandations personnalisées")
                
                # Système d'onglets pour les recommandations
                rec_tab1, rec_tab2, rec_tab3 = st.tabs(["Suivi médical", "Mode de vie", "Surveillance"])
                
                with rec_tab1:
                    st.markdown("""
                    ### Suivi médical recommandé
                    
                    **Rappel**: Ces recommandations sont générales et doivent être adaptées par votre médecin.
                    
                    #### Fréquence des visites médicales:
                    """)
                    
                    # Recommandations adaptées au niveau de risque
                    if risk_percentage > 70:
                        st.warning("""
                        ⚠️ **Risque élevé** - Suivi recommandé:
                        - Consultation cardiologique dans les meilleurs délais
                        - Examens complémentaires à envisager: test d'effort, échocardiographie, scanner coronaire
                        - Suivi trimestriel avec votre cardiologue
                        - Surveillance mensuelle de la tension artérielle
                        """)
                    elif risk_percentage > 30:
                        st.info("""
                        ⚠️ **Risque modéré** - Suivi recommandé:
                        - Consultation médicale dans les 3 mois
                        - Bilan cardiologique annuel
                        - Surveillance trimestrielle de la tension artérielle
                        - Évaluation des facteurs de risque tous les 6 mois
                        """)
                    else:
                        st.success("""
                        ✅ **Risque faible** - Suivi recommandé:
                        - Visite médicale annuelle de routine
                        - Bilan cardiaque préventif tous les 2 ans
                        - Surveillance régulière de la tension artérielle (tous les 6 mois)
                        """)
                
                with rec_tab2:
                    st.markdown("""
                    ### Recommandations sur le mode de vie
                    
                    Une alimentation équilibrée et l'activité physique sont essentielles pour maintenir un cœur sain.
                    """)
                    
                    # Deux colonnes pour les recommandations
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.markdown("""
                        #### 🥗 Alimentation
                        
                        - **Privilégiez**: fruits, légumes, céréales complètes, légumineuses
                        - **Limitez**: sel (< 5g/jour), sucres ajoutés, graisses saturées
                        - **Évitez**: aliments ultra-transformés, charcuteries
                        - **Hydratation**: 1,5 à 2L d'eau par jour
                        - **Modèle recommandé**: régime méditerranéen
                        """)
                    
                    with col2:
                        st.markdown("""
                        #### 🏃 Activité physique
                        
                        - **Fréquence**: 30 minutes d'activité modérée, 5 fois par semaine
                        - **Types d'exercices**: marche rapide, natation, vélo, jardinage
                        - **Intensité**: adaptée à votre condition (parler sans essoufflement)
                        - **Renforcement musculaire**: 2 séances par semaine
                        - **Réduction du temps assis**: pauses actives toutes les heures
                        """)
                    
                    # Recommandations spécifiques selon les facteurs de risque
                    specific_recs = []
                    
                    if any(factor == "Cholestérol élevé" for factor, _, _, _ in risk_factors):
                        specific_recs.append("""
                        **Pour le cholestérol élevé**:
                        - Réduire les graisses saturées (viandes grasses, produits laitiers entiers)
                        - Augmenter la consommation de fibres solubles (avoine, légumineuses)
                        - Consommer des oméga-3 (poissons gras, noix)
                        """)
                    
                    if any(factor == "Hypertension" for factor, _, _, _ in risk_factors):
                        specific_recs.append("""
                        **Pour l'hypertension**:
                        - Réduire drastiquement le sel (lire les étiquettes)
                        - Privilégier les aliments riches en potassium (bananes, épinards)
                        - Pratiquer des exercices d'endurance réguliers
                        - Techniques de gestion du stress (méditation, yoga)
                        """)
                    
                    if any(factor == "Glycémie élevée" for factor, _, _, _ in risk_factors):
                        specific_recs.append("""
                        **Pour la glycémie élevée**:
                        - Réduire les sucres simples et les aliments à index glycémique élevé
                        - Favoriser les repas réguliers en petites quantités
                        - Activité physique après les repas
                        - Privilégier les fibres qui ralentissent l'absorption du glucose
                        """)
                    
                    if specific_recs:
                        st.markdown("#### Recommandations spécifiques à vos facteurs de risque")
                        for rec in specific_recs:
                            st.markdown(rec)
                
                with rec_tab3:
                    st.markdown("""
                    ### Surveillance régulière recommandée
                    
                    Un suivi régulier de vos paramètres de santé est essentiel pour prévenir les complications.
                    """)
                    
                    # Tableau de fréquence de surveillance
                    surveillance_data = {
                        "Paramètre": ["Tension artérielle", "Cholestérol", "Glycémie", "Électrocardiogramme", "Test d'effort"],
                        "Fréquence": [
                            "Mensuelle" if risk_percentage > 70 else "Trimestrielle" if risk_percentage > 30 else "Semestrielle",
                            "Tous les 3 mois" if risk_percentage > 70 else "Tous les 6 mois" if risk_percentage > 30 else "Annuelle",
                            "Tous les 3 mois" if risk_percentage > 70 else "Tous les 6 mois" if risk_percentage > 30 else "Annuelle",
                            "Tous les 6 mois" if risk_percentage > 70 else "Annuelle" if risk_percentage > 30 else "Tous les 2 ans",
                            "Annuel" if risk_percentage > 70 else "Tous les 2 ans" if risk_percentage > 30 else "Selon avis médical"
                        ],
                        "Valeurs cibles": [
                            "< 140/90 mmHg (< 130/80 si diabète)",
                            "LDL < 100 mg/dL, HDL > 40 mg/dL (H) ou > 50 mg/dL (F)",
                            "À jeun < 100 mg/dL, HbA1c < 5.7%",
                            "Rythme sinusal normal",
                            "Capacité d'effort adaptée à l'âge"
                        ]
                    }
                    
                    st.table(pd.DataFrame(surveillance_data))
                    
                    # Conseil pour l'auto-surveillance
                    st.info("""
                    **💡 Conseil**: Tenez un journal de santé pour suivre vos paramètres et partagez-le avec votre médecin lors des consultations.
                    Des applications mobiles peuvent vous aider à suivre ces paramètres quotidiennement.
                    """)
                
                # Section pour le téléchargement du rapport
                st.subheader("📥 Télécharger le rapport complet")
                
                # Préparation des données pour le rapport
                report_data = {
                    "Date d'analyse": datetime.now().strftime("%d/%m/%Y %H:%M"),
                    "Âge": age,
                    "Sexe": "Homme" if sex == 1 else "Femme",
                    "Type de douleur thoracique": get_label_from_dict(cp_dict, cp),
                    "Tension artérielle": f"{trestbps} mm Hg",
                    "Cholestérol": f"{chol} mg/dl",
                    "Glycémie à jeun > 120 mg/dl": "Oui" if fbs == 1 else "Non",
                    "ECG au repos": get_label_from_dict(restecg_dict, restecg),
                    "Fréquence cardiaque max": thalach,
                    "Angine induite par exercice": "Oui" if exang == 1 else "Non",
                    "Dépression ST": oldpeak,
                    "Pente segment ST": get_label_from_dict(slope_dict, slope),
                    "Nombre de vaisseaux colorés": ca,
                    "Thalassémie": get_label_from_dict(thal_dict, thal),
                    "Prédiction": "Risque de maladie cardiaque" if prediction == 1 else "Faible risque de maladie cardiaque",
                    "Probabilité de risque": f"{risk_percentage:.1f}%",
                    "Niveau de risque": risk_level.capitalize()
                }
                
                # Convertir en DataFrame pour CSV
                report_df = pd.DataFrame([report_data])
                
                # Générer le CSV
                csv = report_df.to_csv(index=False)
                
                # Téléchargement du rapport
                st.download_button(
                    label="📋 Télécharger le rapport d'analyse (CSV)",
                    data=csv,
                    file_name=f"rapport_cardiaque_{datetime.now().strftime('%Y%m%d_%H%M')}.csv",
                    mime="text/csv",
                    help="Télécharger les résultats et les données de l'analyse dans un fichier CSV"
                )
                
 
def generate_pdf_report(patient_data, prediction, probability, risk_level, risk_factors):
    """
    Générer un rapport PDF détaillé avec les données du patient et les résultats d'analyse
    Avec correction pour les caractères Unicode
    """
    pdf = FPDF()
    # Définir utf8 comme encodage par défaut
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.add_page()
    
    # En-tête du rapport
    pdf.set_font('Arial', 'B', 16)
    pdf.cell(190, 10, 'RAPPORT D\'ANALYSE DU RISQUE CARDIAQUE', 0, 1, 'C')
    pdf.line(10, 20, 200, 20)
    pdf.ln(5)
    
    # Date et heure
    pdf.set_font('Arial', '', 10)
    pdf.cell(190, 10, f"Date d'analyse: {datetime.now().strftime('%d/%m/%Y %H:%M')}", 0, 1, 'R')
    pdf.ln(5)
    
    # Données du patient
    pdf.set_font('Arial', 'B', 12)
    pdf.cell(190, 8, 'Informations du patient', 0, 1, 'L')
    pdf.ln(2)
    
    # Création d'un tableau pour les données
    pdf.set_font('Arial', '', 10)
    pdf.set_fill_color(240, 240, 240)
    
    # Première colonne
    col_width = 95
    row_height = 8
    
    # En-têtes des colonnes
    pdf.set_font('Arial', 'B', 10)
    pdf.cell(col_width, row_height, 'Parametre', 1, 0, 'L', 1)
    pdf.cell(col_width, row_height, 'Valeur', 1, 1, 'L', 1)
    
    # Données démographiques - traiter chaque champ individuellement pour éviter les problèmes d'encodage
    pdf.set_font('Arial', '', 10)
    
    # Liste des paramètres sans accent pour éviter les problèmes d'encodage
    params_mapping = {
        'Âge': 'Age',
        'Sexe': 'Sexe',
        'Type de douleur thoracique': 'Type de douleur thoracique',
        'Tension artérielle': 'Tension arterielle',
        'Cholestérol': 'Cholesterol',
        'Glycémie à jeun > 120 mg/dl': 'Glycemie a jeun > 120 mg/dl',
        'ECG au repos': 'ECG au repos',
        'Fréquence cardiaque max': 'Frequence cardiaque max',
        'Angine induite par exercice': 'Angine induite par exercice',
        'Dépression ST': 'Depression ST',
        'Pente segment ST': 'Pente segment ST',
        'Nombre de vaisseaux colorés': 'Nombre de vaisseaux colores',
        'Thalassémie': 'Thalassemie'
    }
    
    # Parcourir les données du patient et les afficher
    for original_key, display_key in params_mapping.items():
        # Vérifier si la clé existe dans patient_data
        if original_key in patient_data:
            value = patient_data[original_key]
            # Conversion des valeurs en chaînes de caractères
            if not isinstance(value, str):
                value = str(value)
            
            pdf.cell(col_width, row_height, display_key, 1, 0, 'L')
            pdf.cell(col_width, row_height, value, 1, 1, 'L')
    
    pdf.ln(10)
    
    # Résultats de l'analyse
    pdf.set_font('Arial', 'B', 14)
    pdf.cell(190, 10, 'Resultats de l\'analyse', 0, 1, 'L')
    
    # Cadre pour le résultat principal
    pdf.set_font('Arial', 'B', 12)
    if prediction == 1:
        pdf.set_fill_color(255, 240, 240)  # Rouge clair
        pdf.cell(190, 12, f'ATTENTION: Risque {risk_level} de maladie cardiaque', 1, 1, 'C', 1)
    else:
        pdf.set_fill_color(240, 255, 240)  # Vert clair
        pdf.cell(190, 12, 'FAVORABLE: Risque faible de maladie cardiaque', 1, 1, 'C', 1)
    
    pdf.ln(5)
    
    # Détails du risque
    pdf.set_font('Arial', '', 10)
    pdf.cell(190, 8, f"Probabilite de risque: {probability:.1f}%", 0, 1, 'L')
    pdf.cell(190, 8, f"Niveau de risque: {risk_level.capitalize()}", 0, 1, 'L')
    
    pdf.ln(5)
    
    # Facteurs de risque identifiés
    if risk_factors:
        pdf.set_font('Arial', 'B', 12)
        pdf.cell(190, 10, 'Facteurs de risque identifies', 0, 1, 'L')
        
        # En-têtes du tableau de facteurs
        pdf.set_font('Arial', 'B', 10)
        pdf.set_fill_color(240, 240, 240)
        pdf.cell(60, row_height, 'Facteur de risque', 1, 0, 'L', 1)
        pdf.cell(30, row_height, 'Valeur', 1, 0, 'L', 1)
        pdf.cell(100, row_height, 'Explication', 1, 1, 'L', 1)
        
        # Données des facteurs
        pdf.set_font('Arial', '', 9)
        for factor, value, explanation, _ in risk_factors:
            # Convertir les valeurs en chaînes de caractères
            if not isinstance(value, str):
                value = str(value)
            # Remplacer les accents
            factor_clean = (factor.replace('é', 'e').replace('è', 'e').replace('à', 'a')
                           .replace('ê', 'e').replace('ô', 'o').replace('ù', 'u'))
            value_clean = value.replace('é', 'e').replace('è', 'e').replace('à', 'a') if isinstance(value, str) else value
            explanation_clean = (explanation.replace('é', 'e').replace('è', 'e').replace('à', 'a')
                               .replace('ê', 'e').replace('ô', 'o').replace('ù', 'u'))
            
            # Gestion des explications longues avec word wrap
            pdf.cell(60, row_height, factor_clean, 1, 0, 'L')
            pdf.cell(30, row_height, value_clean, 1, 0, 'L')
            
            # Pour les explications longues, tronquer l'explication
            if len(explanation_clean) > 60:
                explanation_clean = explanation_clean[:57] + "..."
            pdf.cell(100, row_height, explanation_clean, 1, 1, 'L')
    else:
        pdf.set_font('Arial', '', 10)
        pdf.cell(190, 8, "Aucun facteur de risque majeur identifie", 0, 1, 'L')
    
    pdf.ln(10)
    
    # Recommandations générales - CORRECTION: remplacer les puces Unicode par des tirets
    pdf.set_font('Arial', 'B', 12)
    pdf.cell(190, 10, 'Recommandations generales', 0, 1, 'L')
    
    pdf.set_font('Arial', '', 10)
    
    # Utiliser des tirets (-) au lieu des puces (•)
    if probability > 70:
        pdf.multi_cell(190, 6, "- Consultation cardiologique dans les meilleurs delais\n- Examens complementaires a envisager: test d'effort, echocardiographie\n- Suivi trimestriel avec votre cardiologue\n- Surveillance mensuelle de la tension arterielle", 0, 'L')
    elif probability > 30:
        pdf.multi_cell(190, 6, "- Consultation medicale dans les 3 mois\n- Bilan cardiologique annuel\n- Surveillance trimestrielle de la tension arterielle\n- Evaluation des facteurs de risque tous les 6 mois", 0, 'L')
    else:
        pdf.multi_cell(190, 6, "- Visite medicale annuelle de routine\n- Bilan cardiaque preventif tous les 2 ans\n- Surveillance reguliere de la tension arterielle (tous les 6 mois)", 0, 'L')
    
    pdf.ln(5)
    
    # Note de bas de page
    pdf.set_font('Arial', 'I', 8)
    pdf.set_text_color(100, 100, 100)
    pdf.cell(0, 10, "Ce rapport est genere automatiquement et doit etre interprete par un professionnel de sante.", 0, 1, 'C')
    pdf.cell(0, 10, "Page 1/1", 0, 0, 'C')
    
    # Retourner le PDF comme bytes
    try:
        return pdf.output(dest='S').encode('latin1')
    except UnicodeEncodeError:
        try:
            # En cas d'erreur d'encodage, on essaie une autre méthode
            return pdf.output(dest='S')
        except Exception as e:
            st.error(f"Erreur lors de la génération finale du PDF: {str(e)}")
            return None

def try_generate_pdf(data, prediction_result=None, probability_value=None, risk_factors_list=None):
    """
    Fonction de sécurité pour générer le PDF seulement si toutes les données sont disponibles
    """
    try:
        # Vérification que les variables nécessaires sont définies
        if prediction_result is None or probability_value is None:
            st.error("Données insuffisantes pour générer le rapport PDF.")
            return None
            
        # Détermination du niveau de risque
        if probability_value >= 70:
            risk_level = "eleve"  # Sans accent pour éviter les problèmes d'encodage
        elif probability_value >= 30:
            risk_level = "modere"  # Sans accent
        else:
            risk_level = "faible"
            
        # Générer le rapport
        return generate_pdf_report(
            data,                  # Données du patient
            prediction_result,     # Prédiction (0 ou 1)
            probability_value,     # Probabilité en pourcentage
            risk_level,            # Niveau de risque (faible, modéré, élevé)
            risk_factors_list or []  # Liste des facteurs de risque ou liste vide
        )
    except Exception as e:
        # En cas d'erreur, on affiche un message d'erreur détaillé
        st.error(f"Erreur lors de la génération du PDF: {str(e)}")
        import traceback
        st.error(f"Détails: {traceback.format_exc()}")
        return None

# Exemple d'utilisation dans une application Streamlit
if 'predict_button' in locals() and predict_button:
    # Votre code de prédiction...
    if 'prediction' in locals() and 'probability' in locals() and prediction is not None and probability is not None:
        # Exemple de format pour report_data et risk_factors
        if 'report_data' not in locals():
            st.error("Les données du rapport ne sont pas disponibles.")
        else:
            # Calculer risk_percentage
            risk_percentage = probability * 100 if 'risk_percentage' not in locals() else risk_percentage
            
            # Obtenir risk_factors
            if 'risk_factors' not in locals():
                risk_factors = []  # Valeur par défaut si non définie
            
            # Appeler la fonction sécurisée
            pdf_bytes = try_generate_pdf(
                report_data,            
                prediction,             
                risk_percentage,        
                risk_factors            
            )
            
            # Afficher le bouton de téléchargement uniquement si pdf_bytes existe
            if pdf_bytes:
                st.download_button(
                    label="📄 Télécharger le rapport détaillé (PDF)",
                    data=pdf_bytes,
                    file_name=f"rapport_cardiaque_detaille_{datetime.now().strftime('%Y%m%d_%H%M')}.pdf",
                    mime="application/pdf",
                    help="Télécharger un rapport détaillé avec visualisations et recommandations au format PDF"
                )
            else:
                st.error("La génération du rapport PDF a échoué. Veuillez réessayer.")
    else:
        st.error("Prédiction ou probabilité non disponible. Impossible de générer le rapport.")
  
    st.markdown("""
    <div style="background-color: #FFF3CD; padding: 15px; border-radius: 10px; 
               margin-top: 30px; border-left: 5px solid #FFC107;">
        <h3 style="color: #856404;">⚠️ Avertissement médical important</h3>
        <p>Ce résultat est basé sur une analyse statistique et ne constitue pas un diagnostic médical.
        Il est impératif de consulter un professionnel de santé qualifié pour une évaluation complète
        et des recommandations personnalisées.</p>
    </div>
    """, unsafe_allow_html=True)
            
    # Suggestions de suivi
    st.subheader("🔄 Que faire maintenant?")
    
    suggestions = [
        "👨‍⚕️ Partagez ces résultats avec votre médecin traitant lors de votre prochaine consultation",
        "📊 Utilisez les informations des facteurs de risque pour améliorer votre mode de vie",
        "📅 Programmez les examens de suivi recommandés selon votre niveau de risque",
        "📱 Envisagez d'utiliser une application de suivi de santé cardiaque pour surveiller vos progrès"
    ]
    
    for suggestion in suggestions:
        st.markdown(f"- {suggestion}")
    
    # Section FAQ
    with st.expander("❓ Questions fréquentes sur l'analyse de risque cardiaque"):
        st.markdown("""
        **Q: Quelle est la précision de cette analyse?**  
        R: Le modèle utilisé a une précision d'environ 85% basée sur des données d'entraînement. Cependant, il ne remplace pas l'évaluation clinique par un médecin.
        
        **Q: Puis-je me fier uniquement à ce résultat?**  
        R: Non, cet outil est conçu pour la sensibilisation et l'éducation. Une évaluation médicale complète est nécessaire pour un diagnostic précis.
        
        **Q: À quelle fréquence devrais-je refaire ce test?**  
        R: Pour un suivi optimal, une évaluation annuelle est recommandée pour les personnes à faible risque, et tous les 6 mois pour celles à risque modéré ou élevé.
        
        **Q: Comment puis-je améliorer mon score?**  
        R: Concentrez-vous sur les facteurs de risque modifiables identifiés dans l'analyse. L'adoption d'un mode de vie sain, incluant une alimentation équilibrée, de l'exercice régulier et l'arrêt du tabac peut significativement réduire votre risque cardiaque.
        """)
    
    # Références et ressources
    with st.expander("📚 Références et ressources utiles"):
        st.markdown("""
        ### Pour en savoir plus sur les maladies cardiovasculaires
        
        #### Associations et organismes officiels
        - [Fédération Française de Cardiologie](https://www.fedecardio.org/)
        - [Société Française de Cardiologie](https://www.sfcardio.fr/)
        - [Association Française des Diabétiques](https://www.federationdesdiabetiques.org/)
        - [Santé Publique France - Maladies cardiovasculaires](https://www.santepubliquefrance.fr/maladies-et-traumatismes/maladies-cardiovasculaires-et-accident-vasculaire-cerebral)
        
        #### Ressources marocaines
        - [Société Marocaine de Cardiologie](https://www.smcmaroc.org/)
        - [Ministère de la Santé du Maroc](https://www.sante.gov.ma/)
        - [Fondation Lalla Salma](https://www.contrelecancer.ma/)
        - [CHU Mohammed VI de Marrakech](https://www.chumarrakech.ma/)
        - [Centre de Cardiologie de Casablanca](https://cardiocasa.ma/)
        - [Association Marocaine de Lutte Contre les Maladies Cardiovasculaires](http://www.amcv.ma/)
        
        #### Ressources éducatives
        - [Ameli Santé - Comprendre les maladies cardiovasculaires](https://www.ameli.fr/assure/sante/themes/risque-cardiovasculaire/definition-facteurs-risque)
        - [INSERM - Dossier sur les maladies cardiovasculaires](https://www.inserm.fr/information-en-sante/dossiers-information/maladies-cardiovasculaires)
        - [Pas à Pas - Programme d'accompagnement du patient cardiaque](https://www.pasapas-sfc.com/)
        
        #### Applications mobiles recommandées
        - "CardioCompanion" - Suivi de la tension artérielle et des facteurs de risque
        - "HeartGuard" - Journal de santé cardiaque avec rappels de médicaments
        - "CardioTrack" - Suivi d'activité physique adapté aux patients cardiaques
        """)

else:
    # st.info("👆 Cliquez sur le bouton 'Analyser les données et prédire le risque' pour lancer l'analyse.")
    
    # Afficher des informations générales sur les maladies cardiaques
    with st.expander("ℹ️ Informations sur les maladies cardiovasculaires"):
        st.markdown("""
        ### Les maladies cardiovasculaires en chiffres
        
        Les maladies cardiovasculaires représentent la première cause de mortalité dans le monde avec plus de 17 millions de décès par an, soit 31% de la mortalité mondiale totale.
        
        En France, elles sont responsables d'environ 140 000 décès par an et constituent:
        - La 1ère cause de mortalité chez les femmes
        - La 2ème cause de mortalité chez les hommes
        
        ### Facteurs de risque principaux
        
        **Non modifiables:**
        - Âge (risque augmente avec l'âge)
        - Sexe (les hommes sont plus à risque jusqu'à 55 ans, puis le risque s'équilibre)
        - Antécédents familiaux
        
        **Modifiables:**
        - Tabagisme
        - Hypertension artérielle
        - Taux de cholestérol élevé
        - Diabète
        - Surpoids et obésité
        - Sédentarité
        - Stress chronique
        - Alimentation déséquilibrée
        
        ### Prévention
        
        La bonne nouvelle est que 80% des maladies cardiovasculaires pourraient être évitées grâce à:
        - L'adoption d'une alimentation équilibrée
        - La pratique régulière d'une activité physique
        - L'arrêt du tabac
        - La limitation de la consommation d'alcool
        - Le contrôle régulier de la tension artérielle, du cholestérol et de la glycémie
        """)
    
    # Afficher un exemple de résultat
    with st.expander("🔍 Voir un exemple de résultat d'analyse"):
        st.image("https://via.placeholder.com/800x400?text=Exemple+d'analyse+de+risque+cardiaque", 
                caption="Exemple de visualisation des résultats", use_column_width=True)
        st.caption("L'image ci-dessus est un exemple illustratif des résultats que vous obtiendrez après l'analyse.")