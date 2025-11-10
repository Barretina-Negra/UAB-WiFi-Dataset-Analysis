import streamlit as st
import requests
from dotenv import load_dotenv
import os

# Load environment variables
load_dotenv()

# --- Configuració de la pàgina ---
st.set_page_config(page_title="AINA Chatbot", page_icon="💬", layout="centered")

st.title("💬 AINA Chatbot (PublicAI)")
st.write("Fes una pregunta al model **ALIA-40B-Instruct** d'AINA i obtén una resposta en català.")

# --- Configuració de la API ---
API_KEY = os.getenv("AINA_API_KEY")
if not API_KEY:
    st.error("❌ AINA_API_KEY no trobada a les variables d'entorn. Si us plau, crea un fitxer .env amb AINA_API_KEY=tu_api_key")
    st.stop()

headers = {
    "Authorization": f"Bearer {API_KEY}",
    "Content-Type": "application/json",
    "User-Agent": "UAB-THE-HACK/1.0"
}

# --- Input de l'usuari ---
user_input = st.text_area("✏️ Escriu la teva pregunta:",
                          placeholder="Exemple: Com puc crear un chatbot amb Python?")

if st.button("Enviar"):
    if not user_input.strip():
        st.warning("⚠️ Escriu una pregunta abans d'enviar.")
    else:
        with st.spinner("Esperant resposta d’AINA..."):
            # Cos de la petició
            payload = {
                "model": "BSC-LT/ALIA-40b-instruct_Q8_0",
                "messages": [{"role": "user", "content": user_input}],
                "max_tokens": 500,
                "temperature": 0.7,
            }

            response = requests.post(
                "https://api.publicai.co/v1/chat/completions",
                headers=headers,
                json=payload
            )

            # Mostrar resultat
            if response.status_code == 200:
                data = response.json()
                resposta = data["choices"][0]["message"]["content"]
                st.success("**Resposta d’AINA:**")
                st.write(resposta)
            else:
                st.error(f"Error {response.status_code}: {response.text}")
