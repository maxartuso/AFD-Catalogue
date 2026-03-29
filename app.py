import streamlit as st
import pandas as pd
import openai
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
import os

# --- 1. INITIALISATION ---

# --- CONFIGURATION DES SECRETS ---
# Streamlit récupère automatiquement la clé dans .streamlit/secrets.toml
if "OPENAI_API_KEY" in st.secrets:
    os.environ["OPENAI_API_KEY"] = st.secrets["OPENAI_API_KEY"]
    client = openai.OpenAI(api_key=st.secrets["OPENAI_API_KEY"])
else:
    st.error("Clé API manquante ! Ajoutez-la dans le fichier secrets.toml ou dans les paramètres Streamlit.")
    st.stop()

st.set_page_config(page_title="IA Catalogue AFD", page_icon="🧩")
st.title("🧩 Assistant Conseil Autisme Diffusion")


# --- 2. MOTEUR DE RECHERCHE (CACHE) ---
@st.cache_resource
def preparer_moteur_recherche():
    # Chargement du fichier
    if not os.path.exists("catalogue_afd.csv"):
        st.error("Fichier 'catalogue_afd.csv' introuvable !")
        return None, None, None

    df = pd.read_csv("catalogue_afd.csv").fillna("")

    # On crée le texte que l'IA va analyser pour comparer les produits
    df['text_complet'] = (
            "Produit: " + df['nom'] +
            " | Catégorie: " + df['categorie'] +
            " | Description: " + df['description']
    )

    # Transformation des textes en vecteurs (Embeddings gratuits et locaux)
    model = SentenceTransformer('all-MiniLM-L6-v2')
    embeddings = model.encode(df['text_complet'].tolist(), convert_to_tensor=False)

    # Création de l'index FAISS (Recherche ultra-rapide)
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index.add(np.array(embeddings).astype('float32'))

    return df, model, index


# --- 3. LOGIQUE DE L'APPLICATION ---
df, model, index = preparer_moteur_recherche()

if df is not None:
    query = st.text_input("Posez votre question (ex: 'Que proposez-vous pour l'aide à l'habillage ?')")

    if query:
        with st.spinner("Recherche des meilleures solutions..."):
            # A. Trouver les 3 produits les plus pertinents
            query_vector = model.encode([query])
            distances, indices = index.search(np.array(query_vector).astype('float32'), k=3)

            # B. Construire la liste des produits pour l'IA
            contexte_produits = ""
            for i in indices[0]:
                p = df.iloc[i]
                contexte_produits += f"- {p['nom']} (Catégorie: {p['categorie']})\n  Description: {p['description']}\n  Lien: {p['URL']}\n\n"

            # C. Créer le message pour GPT-4 (Le fameux "Prompt")
            prompt_systeme = f"""Tu es l'expert conseil d'Autisme Diffusion (AFD).
            Ta mission est d'aider les parents et professionnels à trouver le matériel adapté.
            Réponds de façon bienveillante en te basant UNIQUEMENT sur ces produits du catalogue :

            {contexte_produits}

            Si aucun produit ne correspond vraiment, suggère de contacter directement AFD.
            Mentionne toujours les noms des produits et affiche les liens URL pour qu'ils soient cliquables."""

            # D. Appel direct à OpenAI
            try:
                response = client.chat.completions.create(
                    model="gpt-5.2",
                    messages=[
                        {"role": "system", "content": "Tu es un conseiller expert en autisme."},
                        {"role": "user", "content": prompt_systeme}
                    ],
                    temperature=0.2
                )

                # Affichage du résultat
                st.markdown("---")
                st.markdown("### 💡 Nos recommandations :")
                st.write(response.choices[0].message.content)

            except Exception as e:
                st.error(f"Erreur API OpenAI : {e}")
