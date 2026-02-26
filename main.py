
import streamlit as st
import logging
import os
import tempfile
import shutil
from groq import Groq
import warnings

# Supprimer les avertissements de torch
warnings.filterwarnings('ignore', category=UserWarning, message='.*torch.classes.*')

from langchain_community.document_loaders import PyPDFLoader
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain.prompts import ChatPromptTemplate, PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_groq import ChatGroq
from langchain_core.runnables import RunnablePassthrough
from langchain.retrievers.multi_query import MultiQueryRetriever
from typing import List, Tuple, Any, Optional, Generator

# Configuration de la page Streamlit
st.set_page_config(
    page_title="CCG genIA",
    page_icon="🎈",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# Configuration du logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)

logger = logging.getLogger(__name__)

# Initialisation du client Groq
client = Groq(api_key=st.secrets["GROQ_API_KEY"])

@st.cache_resource
def get_embedding_model():
    """Charge le modèle d'embedding HuggingFace et le met en cache."""
    logger.info("Chargement du modèle d'embedding...")
    return HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

@st.cache_data
def create_vector_db(_file_upload) -> FAISS:
    """Crée une base de données vectorielle à partir d'un fichier PDF téléchargé."""
    logger.info(f"Création de la base de données vectorielle à partir de : {_file_upload.name}")
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmpfile:
        tmpfile.write(_file_upload.getvalue())
        tmp_path = tmpfile.name
        logger.info(f"Fichier temporaire créé à : {tmp_path}")

    loader = PyPDFLoader(tmp_path)
    data = loader.load()
    
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=7500, chunk_overlap=100)
    chunks = text_splitter.split_documents(data)
    logger.info("Document divisé en chunks")

    embeddings = get_embedding_model()
    vector_db = FAISS.from_documents(documents=chunks, embedding=embeddings)
    logger.info("Base de données vectorielle créée avec FAISS")

    os.remove(tmp_path)
    logger.info(f"Fichier temporaire {tmp_path} supprimé")
    return vector_db

def process_question(question: str, vector_db: FAISS, selected_model: str) -> str:
    """Traite une question de l'utilisateur en utilisant la base de données vectorielle et le modèle de langage sélectionné."""
    logger.info(f"Traitement de la question : {question} avec le modèle : {selected_model}")
    
    llm = ChatGroq(model=selected_model, groq_api_key=st.secrets["GROQ_API_KEY"])
    
    QUERY_PROMPT = PromptTemplate(
        input_variables=["question"],
        template="""Vous êtes un assistant modèle linguistique IA.
        Votre tâche est de générer 2 versions différentes de la question
        de l'utilisateur afin de récupérer des documents pertinents dans une base de
        données vectorielle. En générant plusieurs perspectives sur la question de l'utilisateur,
        votre objectif est d'aider l'utilisateur à surmonter certaines des limitations 
        de la recherche basée sur la similarité de distance.
        Fournissez ces questions alternatives séparées par des sauts de ligne.
        Question originale : {question}""",
    )

    retriever = MultiQueryRetriever.from_llm(
        vector_db.as_retriever(), 
        llm,
        prompt=QUERY_PROMPT
    )

    template = """Répondez à la question en vous basant UNIQUEMENT sur le contexte suivant:
    {context}
    Question: {question}
    """

    prompt = ChatPromptTemplate.from_template(template)

    chain = (
        {"context": retriever, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )

    response = chain.invoke(question)
    logger.info("Question traitée et réponse générée")
    return response

def generate_chat_responses(chat_completion) -> Generator[str, None, None]:
    """Génère le contenu de la réponse du chat à partir de la réponse de l'API Groq."""
    for chunk in chat_completion:
        if chunk.choices[0].delta.content:
            yield chunk.choices[0].delta.content

def process_general_question(question: str, selected_model: str) -> str:
    """Traite une question de l'utilisateur en utilisant le client Groq et le modèle de langage sélectionné."""
    logger.info(f"Traitement de la question : {question} avec le modèle : {selected_model}")
    full_response = "error: nothing returned"
    try:
        st.session_state.messages.append({"role": "user", "content": question})

        chat_completion = client.chat.completions.create(
            model=selected_model,
            messages=[
                {"role": m["role"], "content": m["content"]}
                for m in st.session_state.messages
            ],
            stream=True
        )

        chat_responses_generator = generate_chat_responses(chat_completion)
        full_response = st.write_stream(chat_responses_generator)

        st.session_state.messages.append({"role": "assistant", "content": full_response})

    except Exception as e:
        st.error(e, icon="🚨")

    logger.info("Question traitée et réponse générée")
    return full_response

def delete_vector_db() -> None:
    """Supprime la base de données vectorielle et efface l'état de la session associé."""
    logger.info("Suppression de la base de données vectorielle")
    st.session_state.pop("vector_db", None)
    st.session_state.pop("file_upload", None)
    st.success("Base de données vectorielle et fichiers temporaires supprimés avec succès.")
    logger.info("Base de données vectorielle et état de la session associé effacés")
    st.rerun()

def main() -> None:
    """Fonction principale pour exécuter l'application Streamlit."""
    st.subheader("🧠 CCG RAG Model", divider="gray", anchor=False)
    
    if "messages" not in st.session_state:
        st.session_state.messages = []

    if "selected_model" not in st.session_state:
        st.session_state.selected_model = None

    models = {
        #"gemma2-9b-it": {"name": "Gemma2-9b-it", "tokens": 8192, "developer": "Google"},
        "llama-3.3-70b-versatile": {"name": "LLaMA3.3-70b-versatile", "tokens": 128000, "developer": "Meta"},
        "llama-3.1-8b-instant": {"name": "LLaMA3.1-8b-instant", "tokens": 128000, "developer": "Meta"},
        #"llama3-70b-8192": {"name": "LLaMA3-70b-8192", "tokens": 8192, "developer": "Meta"},
        #"meta-llama/llama-prompt-guard-2-86m": {"name": "LLaMA3-guard", "tokens": 128000, "developer": "Meta"},
        "meta-llama/llama-4-scout-17b-16e-instruct": {"name": "LLaMA3-4-scount", "tokens": 128000, "developer": "Meta"},
        #"llama3-8b-8192": {"name": "LLaMA3-8b-8192", "tokens": 8192, "developer": "Meta"},
        #"mixtral-8x7b-32768": {"name": "Mixtral-8x7b-Instruct-v0.1", "tokens": 32768, "developer": "Mistral"},
    }

    col1, col2 = st.columns([1.5, 2])

    with col1:
        model_option = st.selectbox(
            "Choisissez un modèle:",
            options=list(models.keys()),
            format_func=lambda x: models[x]["name"],
            index=1
        )  

    if st.session_state.selected_model != model_option:
        st.session_state.messages = []
        st.session_state.selected_model = model_option

    selected_model = model_option
        
    file_upload = col1.file_uploader(
        "Téléchargez un fichier PDF ↓", 
        type="pdf", 
        accept_multiple_files=False,
        key="pdf_uploader"
    )

    if file_upload:
        if "vector_db" not in st.session_state or st.session_state.vector_db is None:
            with st.spinner("Traitement du PDF téléchargé..."):
                st.session_state["vector_db"] = create_vector_db(file_upload)
                st.session_state["file_upload"] = file_upload

    if "vector_db" in st.session_state and st.session_state.vector_db is not None:
        delete_collection = col1.button(
            "⚠️ Supprimer la collection", 
            type="secondary",
            key="delete_button"
        )
        if delete_collection:
            delete_vector_db()

    with col2:
        message_container = st.container(height=500, border=True)
    
        for i, message in enumerate(st.session_state["messages"]):
            avatar = "🤖" if message["role"] == "assistant" else "😎"
            with message_container.chat_message(message["role"], avatar=avatar):
                st.markdown(message["content"])
    
        if prompt := st.chat_input("Entrez une requête ici...", key="chat_input"):
            try:
                st.session_state["messages"].append({"role": "user", "content": prompt})
                with message_container.chat_message("user", avatar="😎"):
                    st.markdown(prompt)
    
                with message_container.chat_message("assistant", avatar="🤖"):
                    with st.spinner(":green[traitement...]"):
                        if "vector_db" in st.session_state and st.session_state.vector_db is not None:
                            response = process_question(
                                prompt, st.session_state["vector_db"], selected_model
                            )
                            st.markdown(response)
                        else:
                            response = process_general_question(prompt, selected_model)
            
            except Exception as e:
                st.error(e, icon="⛔️")
                logger.error(f"Erreur lors du traitement de la requête : {e}")
        else:
            if "vector_db" not in st.session_state or st.session_state.vector_db is None:
                st.warning("Veuillez télécharger un fichier PDF pour commencer le chat...")

if __name__ == "__main__":
    main()
