 #!/usr/bin/env python
# coding: utf-8

# # Fichier Exemple : utilisation des modèles développés
# ### 1. Modèle IA pour TC
# ### 2. Modèle IA pour PC
# ### 3. Modèle IA pour ACEN
# ### 4. Modèle IA pour NBP - normal boiling point
# ### 5. Modèle IA pour TTR - point triple
# ### 6. Modèle IA pour VC
# ### Copyright - LRGP - Nancy 2024 - Roda Bounaceur
# 

import pandas as pd

import streamlit as st
import joblib
import numpy as np
import warnings
import os
warnings.simplefilter("ignore")

#lirairies chimie 
from sklearn.base import BaseEstimator,RegressorMixin
from rdkit import Chem 
from rdkit.Chem import AllChem, Descriptors, Draw
import rdkit.Chem.inchi
from mordred import Calculator, descriptors as mordred_descriptors
#streamlit ketcher (dessinateur de molecule )
from streamlit_ketcher import st_ketcher


#configuration de la page 
st.set_page_config(
    page_title="Prédiction des propriétés de composés chimiques - Plateforme IA - LRGP - Nancy",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# 2 style css personnalisé 

st.markdown(
    """
    <style>

    .main-title {
        font-size: 2.2rem;
        font-weight: bold;
        color: #1F4E79;
     
        margin-bottom: 0rem;
    }

  .subtitle {
        font-size: 1rem;
        color: #555;
        margin-bottom: 1.5rem;
    }
    .result-card {
        background-color: #EBF3FB;
        border-left: 5px solid #2E75B6;
        padding: 1rem;
        border-radius: 8px;
        margin-bottom: 1rem;
    }
    .warning-box {
        background-color: #FFF3CD;
        border-left: 5px solid #FFC107;
        padding: 0.8rem;
        border-radius: 6px;
    }
    .footer {
        font-size: 0.8rem;
        color: #999;
        text-align: center;
        margin-top: 3rem;
    }

    </style>
    """,
    unsafe_allow_html=True
)

#classe MetaModel  dans Fichier_Exemple pour charger les modèles de prédiction
#Developé par Roda Bounaceur - LRGP - Nancy 2024
class MetaModel(BaseEstimator, RegressorMixin):
    def __init__(self, models):
        self.models = models
    
    def fit(self, X, y):
        for model in self.models:
            model.fit(X, y)
        return self
    
    def predict(self, X):
        predictions = [model.predict(X) for model in self.models]
        return sum(predictions) / len(self.models)
    


#chargement des modèles IA développés pour la prédiction de TC, PC, ACEN et NBP (mis en cache)
@st.cache_resource (show_spinner="Chargement des modèles IA...")
def load_models():
    #charger les 4 mmodeles IA une suele fois au demarrage de l'application pour optimiser les performances
    #le cache evite de les recharger à chaque interaction.
    base = os.path.dirname(os.path.abspath(__file__))
    TC   = joblib.load(os.path.join(base, '01_modele_final_TC.joblib'))
    PC   = joblib.load(os.path.join(base, '02_modele_final_PC.joblib'))
    ACEN = joblib.load(os.path.join(base, '03_modele_final_ACEN.joblib'))
    NBP  = joblib.load(os.path.join(base, '04_modele_final_NBP.joblib'))

    return TC, PC, ACEN, NBP    


@st.cache_data (show_spinner=False)
def charger_colonnes():
    #charger la liste des descripteurs à conserver pour la prédiction de TC (mis en cache)
    #le cache evite de les relire à chaque interaction.
    """Charge la liste des 247 descripteurs retenus pour TC."""
    base = os.path.dirname(os.path.abspath(__file__))
    with open(os.path.join(base, 'noms_colonnes_247_TC.txt'), 'r') as f:
        cols = [ligne.strip() for ligne in f]
    del cols[0]
    return cols



#fonction pour calculer les descripteurs à partir de la structure moléculaire (SMILES)
def calculer_descripteurs_mordred(smiles_list):    
    """Calcule les descripteurs à partir du SMILES et retourne un DataFrame avec les colonnes spécifiées."""
    calc = Calculator(mordred_descriptors, ignore_3D=True)
    mols = [Chem.MolFromSmiles(smi) for smi in smiles_list]
    return calc.pandas(mols)


def nettoyer_descripteurs(df):
    """Nettoie les descripteurs : remplace les NaN et valeurs non numériques par 0."""
    for col in df.columns:
      df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)
    return df.fillna(0).astype(float)


def valider_smiles(smiles):
    """verifier si un smile est valide avec RDKit."""
    mol= Chem.MolFromSmiles(smiles)
    return mol is not None, mol


def predire_proprietes(smiles):
    """
    Fonction principale : prend un SMILES et retourne
    les 4 propriétés thermodynamiques prédites par les modèles IA.
    """
    TC_model, PC_model, ACEN_model, NBP_model = load_models()
    noms_colonnes = charger_colonnes()
 
    # Calcul des descripteurs Mordred
    df_desc = calculer_descripteurs_mordred([smiles])
 
    # Sélection des 247 descripteurs pertinents
    X = df_desc[noms_colonnes]
 
    # Nettoyage
    X = nettoyer_descripteurs(X)
 
    # Prédictions
    tc   = TC_model.predict(X)[0]
    pc   = PC_model.predict(X)[0]
    acen = ACEN_model.predict(X)[0]
    nbp  = NBP_model.predict(X)[0]
 
    return tc, pc, acen, nbp
 
 # barre laterale pour afficher les instructions et les informations sur l'application
with st.sidebar:
    st.image("https://media.licdn.com/dms/image/v2/D4E0BAQETe5Myk-nlsQ/company-logo_200_200/company-logo_200_200/0/1727789760495/lrgp_nancy_logo?e=2147483647&v=beta&t=ch_7NJa6n6em_OwmOgdOrWuyfe5pSrvTkdimndUDzKk", width=180)
    st.markdown("---")
    st.markdown("### 🔬 À propos")
    st.info(
        "cette plateforme utilise des modèles d'intelligence artificielle développés au LRGP pour prédire" \
        " les propriétés thermodynamiques de composés chimiques " \
        "à partir de leur structure moléculaire (SMILES)."
    )
    st.markdown("---")
    st.markdown("###  Propriétés prédites")
    st.markdown("""
- **TC** — Température critique (K)
- **PC** — Pression critique (bar)
- **ACEN** — Facteur acentrique (-)
- **NBP** — Point d'ébullition normal (K)
    """)
    st.markdown("---")
    st.markdown("###  Exemples de SMILES")
    st.code("CCCC        → Butane")
    st.code("CCCCCC      → Hexane")
    st.code("c1ccccc1    → Benzène")
    st.code("CCO         → Éthanol")
    st.markdown("---")  
    st.markdown(
        "<div class='footer'>LRGP — UMR CNRS 7274<br>ENSIC Nancy — 2025</div>",
        unsafe_allow_html=True
    )

    # entête de la page principale
st.markdown("<div class='main-title'>🔬 Plateforme IA — Propriétés Thermodynamiques</div>", unsafe_allow_html=True)
st.markdown("<div class='subtitle'>LRGP Nancy — Laboratoire Réactions et Génie des Procédés (UMR CNRS 7274)</div>", unsafe_allow_html=True)
st.markdown("---")



#  onglet principale pour la prédiction des propriétés à partir du SMILES
tab1, tab2, tab3 = st.tabs(["🔍 Prédiction à partir du SMILES",
                       "✏️ Dessiner une molécule",
                       "📊 Prédiction par fichier"])



# onglet 1 : prédiction à partir du SMILES
with tab1:
    st.subheader("Prédiction à partir d'une notation SMILES")
    st.write("Entrez la notation SMILES de votre molécule pour obtenir ses propriétés thermodynamiques.")
 
    smiles_input = st.text_input(
        "Notation SMILES",
        placeholder="Ex: CCCC (butane), c1ccccc1 (benzène), CCO (éthanol)",
        help="La notation SMILES est une représentation textuelle de la structure d'une molécule."
    )
 
    col_btn, col_ex = st.columns([1, 3])
    with col_btn:
        predire = st.button("Prédire les propriétés", type="primary", use_container_width=True)
    with col_ex:
        if st.button(" Utiliser un exemple (Hexane)", use_container_width=True):
            smiles_input = "CCCCCC"
            st.rerun()
 
    if predire and smiles_input:
        valide, mol = valider_smiles(smiles_input)
 
        if not valide:
            st.error("❌ SMILES invalide. Vérifiez la notation et réessayez.")
        else:
            with st.spinner("⏳ Calcul des descripteurs et prédiction en cours..."):
                try:
                    tc, pc, acen, nbp = predire_proprietes(smiles_input)
 
                    st.success("✅ Prédiction réussie !")
                    st.markdown("---")
 
                    # Affichage de la molécule
                    col_mol, col_res = st.columns([1, 2])
 
                    with col_mol:
                        st.markdown("#### Structure moléculaire")
                        img = Draw.MolToImage(mol, size=(300, 250))
                        st.image(img, caption=f"SMILES : {smiles_input}")
 
                        # Informations sur la molécule
                        inchikey = rdkit.Chem.inchi.MolToInchiKey(mol)
                        smiles_canon = Chem.MolToSmiles(mol)
                        st.markdown(f"**SMILES canonique :** `{smiles_canon}`")
                        st.markdown(f"**InChIKey :** `{inchikey}`")
 
                    with col_res:
                        st.markdown("#### Propriétés thermodynamiques prédites")
 
                        c1, c2 = st.columns(2)
                        c1.metric(
                            label="🌡️ TC — Température critique",
                            value=f"{tc:.2f} K",
                            delta=f"{tc - 273.15:.2f} °C"
                        )
                        c2.metric(
                            label="💨 PC — Pression critique",
                            value=f"{pc:.2f} bar"
                        )
                        c3, c4 = st.columns(2)
                        c3.metric(
                            label="⚗️ ACEN — Facteur acentrique",
                            value=f"{acen:.4f}"
                        )
                        c4.metric(
                            label="🌡️ NBP — Point d'ébullition",
                            value=f"{nbp:.2f} K",
                            delta=f"{nbp - 273.15:.2f} °C"
                        )
 
                        st.markdown("---")
 
                        # Tableau récapitulatif exportable
                        st.markdown("####  Tableau récapitulatif")
                        df_res = pd.DataFrame({
                            "Propriété": ["TC (K)", "PC (bar)", "ACEN (-)", "NBP (K)"],
                            "Valeur prédite": [f"{tc:.4f}", f"{pc:.4f}", f"{acen:.4f}", f"{nbp:.4f}"],
                            "Description": [
                                "Température critique",
                                "Pression critique",
                                "Facteur acentrique",
                                "Point d'ébullition normal"
                            ]
                        })
                        st.dataframe(df_res, use_container_width=True, hide_index=True)
 
                        # Bouton export CSV
                        csv = df_res.to_csv(index=False).encode('utf-8')
                        st.download_button(
                            label="⬇️ Télécharger les résultats (CSV)",
                            data=csv,
                            file_name=f"proprietes_{smiles_input[:10]}.csv",
                            mime="text/csv"
                        )
 
                except Exception as e:
                    st.error(f"❌ Erreur lors de la prédiction : {e}")
                    st.info("💡 Vérifiez que tous les fichiers .joblib sont bien dans le même dossier que app.py")
 


# ONGLET 2 : Dessinateur de molécule (Ketcher)

with tab2:
    st.subheader("✏️ Dessinez votre molécule")
    st.write("Utilisez l'éditeur ci-dessous pour dessiner une molécule. Le SMILES sera généré automatiquement.")
 
    smiles_ketcher = st_ketcher()
 
    if smiles_ketcher:
        st.markdown(f"**SMILES généré :** `{smiles_ketcher}`")
        if st.button("🚀 Prédire avec cette molécule", type="primary"):
            valide, mol = valider_smiles(smiles_ketcher)
            if valide:
                with st.spinner("Calcul en cours..."):
                    try:
                        tc, pc, acen, nbp = predire_proprietes(smiles_ketcher)
                        st.success("✅ Prédiction réussie !")
                        c1, c2, c3, c4 = st.columns(4)
                        c1.metric("🌡️ TC (K)", f"{tc:.2f}")
                        c2.metric("💨 PC (bar)", f"{pc:.2f}")
                        c3.metric("⚗️ ACEN", f"{acen:.4f}")
                        c4.metric("🌡️ NBP (K)", f"{nbp:.2f}")
                    except Exception as e:
                        st.error(f"Erreur : {e}")
            else:
                st.error("SMILES invalide généré par l'éditeur.")
 
# ONGLET 3 : Prédiction par fichier (plusieurs molécules)

with tab3:
    st.subheader(" Prédiction en lot — plusieurs molécules")
    st.write("Uploadez un fichier texte avec une colonne SMILES pour prédire les propriétés de plusieurs molécules d'un coup.")
 
    st.markdown("""
    <div class='warning-box'>
     <b>Format attendu :</b> fichier .txt ou .csv avec une colonne nommée <code>SMILES</code>,
    séparateur <code>*</code> (comme dans Liste_Alcanes.txt du tuteur)
    </div>
    """, unsafe_allow_html=True)
    st.markdown("")
 
    fichier = st.file_uploader("Choisir un fichier", type=["txt", "csv"])
 
    col_sep1, col_sep2 = st.columns(2)
    with col_sep1:
        separateur = st.selectbox("Séparateur", ["*", ",", ";", "\t"], index=0)
    with col_sep2:
        col_smiles = st.text_input("Nom de la colonne SMILES", value="SMILES")
 
    if fichier:
        try:
            df_input = pd.read_csv(fichier, sep=separateur)
            st.write(f"✅ Fichier chargé : **{len(df_input)} molécules** détectées")
            st.dataframe(df_input.head(5), use_container_width=True)
 
            if st.button(" Lancer la prédiction en lot", type="primary"):
                resultats = []
                barre = st.progress(0, text="Prédiction en cours...")
 
                for i, row in df_input.iterrows():
                    smi = row[col_smiles]
                    valide, _ = valider_smiles(str(smi))
                    if valide:
                        try:
                            tc, pc, acen, nbp = predire_proprietes(str(smi))
                            resultats.append({
                                "SMILES": smi,
                                "TC (K)": round(tc, 4),
                                "PC (bar)": round(pc, 4),
                                "ACEN (-)": round(acen, 4),
                                "NBP (K)": round(nbp, 4),
                                "Statut": "✅ OK"
                            })
                        except:
                            resultats.append({"SMILES": smi, "TC (K)": "-", "PC (bar)": "-",
                                              "ACEN (-)": "-", "NBP (K)": "-", "Statut": "❌ Erreur"})
                    else:
                        resultats.append({"SMILES": smi, "TC (K)": "-", "PC (bar)": "-",
                                          "ACEN (-)": "-", "NBP (K)": "-", "Statut": "❌ SMILES invalide"})
 
                    barre.progress((i + 1) / len(df_input), text=f"Molécule {i+1}/{len(df_input)}")
 
                barre.empty()
                df_resultats = pd.DataFrame(resultats)
                st.success(f"✅ Prédiction terminée pour {len(df_resultats)} molécules !")
                st.dataframe(df_resultats, use_container_width=True, hide_index=True)
 
                csv_out = df_resultats.to_csv(index=False).encode('utf-8')
                st.download_button(
                    label="⬇️ Télécharger tous les résultats (CSV)",
                    data=csv_out,
                    file_name="resultats_prediction_lrgp.csv",
                    mime="text/csv"
                )
 
        except Exception as e:
            st.error(f"Erreur lors de la lecture du fichier : {e}")
            st.info("Vérifiez le séparateur et le nom de la colonne SMILES.")
 
# 
# 9. PIED DE PAGE

st.markdown("---")
st.markdown(
    "<div class='footer'>© 2025 LRGP — Laboratoire Réactions et Génie des Procédés — UMR CNRS 7274 — ENSIC Nancy<br>"
    "Modèles IA développés par Roda Bounaceur — Interface développée dans le cadre d'un stage ingénieur</div>",
    unsafe_allow_html=True
)
 