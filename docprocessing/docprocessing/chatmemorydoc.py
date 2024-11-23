import os
import streamlit as st
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_community.document_loaders import TextLoader
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains.conversational_retrieval.base import ConversationalRetrievalChain
from typing import List

def load_api_key(env_path: str = '../.env') -> str:
    """
    Carga la clave API de OpenAI desde un archivo .env.

    Args:
        env_path (str): Ruta del archivo .env.

    Returns:
        str: Clave API de OpenAI.

    Raises:
        ValueError: Si la clave API no está configurada.
    """
    load_dotenv(env_path)
    api_key = os.getenv('OPENAIKEY')
    if not api_key:
        print("No se encontró la clave API. Configúrala en el archivo .env")
        raise ValueError("Falta la clave de OpenAI.")
    return api_key
def load_document(file_path: str) -> List[str]:
    """
    Carga un documento de texto desde un archivo.

    Args:
        file_path (str): Ruta al archivo de texto.

    Returns:
        List[str]: Lista con el contenido del documento dividido en líneas.
    """
    loader = TextLoader(file_path)
    return loader.load()
def split_document(document: List[str], chunk_size: int = 1000, chunk_overlap: int = 200) -> List[str]:
    """
    Divide un documento en fragmentos (chunks) con solapamiento.

    Args:
        document (List[str]): Documento cargado como lista de líneas.
        chunk_size (int): Tamaño máximo de cada chunk en caracteres.
        chunk_overlap (int): Cantidad de caracteres solapados entre chunks.

    Returns:
        List[str]: Lista de fragmentos del documento.
    """
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", " ", ""]
    )
    return text_splitter.split_documents(document)
def initialize_llm(api_key: str, model_name: str = "gpt-3.5-turbo", temperature: float = 0.0) -> ChatOpenAI:
    """
    Inicializa el modelo de lenguaje ChatOpenAI.

    Args:
        api_key (str): Clave API de OpenAI.
        model_name (str): Nombre del modelo de OpenAI.
        temperature (float): Configuración de creatividad del modelo.

    Returns:
        ChatOpenAI: Instancia configurada del modelo de lenguaje.
    """
    return ChatOpenAI(
        model_name=model_name,
        temperature=temperature,
        openai_api_key=api_key
    )
def initialize_vector_store(chunks: List[str], api_key: str) -> Chroma:
    """
    Inicializa la base de datos vectorial utilizando Chroma y OpenAIEmbeddings.

    Args:
        chunks (List[str]): Lista de fragmentos de texto.
        api_key (str): Clave API de OpenAI.

    Returns:
        Chroma: Base de datos vectorial configurada.
    """
    embeddings = OpenAIEmbeddings(openai_api_key=api_key)
    return Chroma.from_documents(chunks, embeddings)
def initialize_vector_store(chunks: List[str], api_key: str) -> Chroma:
    """
    Inicializa la base de datos vectorial utilizando Chroma y OpenAIEmbeddings.

    Args:
        chunks (List[str]): Lista de fragmentos de texto.
        api_key (str): Clave API de OpenAI.

    Returns:
        Chroma: Base de datos vectorial configurada.
    """
    embeddings = OpenAIEmbeddings(openai_api_key=api_key)
    return Chroma.from_documents(chunks, embeddings)

# Cargar la clave de API
API_KEY = load_api_key()

# Cargar el documento
document: List[str] = load_document('./data/us-constitution.txt')

# Dividir el documento en chunks
chunks = split_document(document)

# Inicializar el modelo de lenguaje
llm = initialize_llm(API_KEY)

# Inicializar la base de datos vectorial
vector_store = initialize_vector_store(chunks, API_KEY)

# Configuramos el recuperador y la cadena de recuperación
retriever = vector_store.as_retriever()

crc_chain = ConversationalRetrievalChain.from_llm(
    llm=llm,
    retriever=retriever
)

# Configuramos la UI
st.title('Chat with Document')
# Entrada de texto del usuario
query = st.text_input('Input your question')

if query:
    # Manejamos la sesión del chat
    if 'history' not in st.session_state:
        st.session_state['history'] = []

    # Ejecutamos la consulta
    response = crc_chain.run({
        'question': query,
        'chat_history': st.session_state['history']
    })

    # Actualizamos el historial
    st.session_state['history'].append((query, response))

    # Mostramos la respuesta
    st.subheader("Respuesta")
    st.write(response)


