import os
import streamlit as st
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_community.document_loaders import TextLoader
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains.retrieval_qa.base import RetrievalQA
from typing import List

# Carga de variables de entorno
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
# Configuramos el UI
st.title('Chat with Document')
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
def handle_query_retrievalQA(query: str, qa_chain: RetrievalQA):
    """
    Maneja la consulta del usuario utilizando la cadena de recuperación.

    Args:
        query (str): Pregunta ingresada por el usuario.
        qa_chain (RetrievalQA): Cadena de recuperación configurada.

    Returns:
        dict: Respuesta generada por el modelo y fragmentos fuente.
    """
    result = qa_chain(query)
    return result
# Cargamos la clave de API
API_KEY: str = load_api_key()
# Cargamos el documento
document: List[str] = load_document('./data/us-constitution.txt')
# Dividimos el documento en chunks
chunks: List[str] = split_document(document)
st.write(f"El documento ha sido dividido en {len(chunks)} fragmentos.")
# Configuramos el modelo de lenguaje
llm: ChatOpenAI = initialize_llm(API_KEY)
# Inicializamos la base de datos vectorial


vector_store: Chroma = initialize_vector_store(chunks, API_KEY)

# Configuramos el recuperador y la cadena de recuperación
retriever = vector_store.as_retriever()
chain: RetrievalQA = RetrievalQA.from_chain_type(
    llm=llm,
    retriever=retriever,
    return_source_documents=True
)

# Obtener la pregunta desde la entrada del usuario
query: str = st.text_input('Input your question')

if query:
    result = handle_query_retrievalQA(query, chain)
    st.subheader("Respuesta")
    st.write(result["result"])

    st.subheader('Fragmentos fuente')
    for doc in result["source_documents"]:
        st.write(doc.page_content)
