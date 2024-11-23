#Importacion de librerias

import os
import streamlit as st                                                                  #Crear la interfaz grafica
from dotenv import load_dotenv                                                          #Leer variables de entorno
from langchain_openai import ChatOpenAI                                                 #Importar el modelo
from langchain_openai import OpenAIEmbeddings                                           #Creacion de embeddings
from langchain_community.document_loaders import (
    TextLoader, PyPDFLoader, Docx2txtLoader
)                                                                                       #Cargar documentos
from langchain_community.vectorstores import Chroma                                     #Generar la base de datos
from langchain.text_splitter import RecursiveCharacterTextSplitter                      #Generar los chuncks
from langchain.chains.conversational_retrieval.base import ConversationalRetrievalChain #Chat con memoria
from typing import List                                                                 #Documentacion de codigo

###########################################
#VARIABLES GLOBALES
PATH_ROOT_DOCUMENTS = './data/'
PATH_DOCUMENT =  './data/us-constitution.txt'
MODEL_LLM = "gpt-3.5-turbo"
PATH_ENV_VARIABLES = '../.env'
API_KEY_ENV = 'OPENAIKEY'
MODEL_TEMPERATURE = 0.0

###########################################
#FUNCIONES GLOBALES
def clear_history()->None:
    '''
    DELETE HISTORY CONVERSATIONS
    '''
    if 'history' in st.session_state:
        del st.session_state['history']

    if 'history' not in st.session_state:
        st.session_state['history'] = []

    print('Exito en la limpieza de historial')
def uploaded_text()->None:
    ''''
    UPLOADED DOCUMENTS
    '''

    file_name = None
    if uploaded_file and add_file:
        with st.spinner('Reading, chunking and embedding file... '):
            bytes_data = uploaded_file.read()
            file_name = os.path.join(PATH_ROOT_DOCUMENTS, uploaded_file.name)
            #Open the file in write binary mode
            with open(file_name, "wb") as f:
                #write binary data in f mean the file
                f.write(bytes_data)
                # Set the loader
                name, extension = os.path.splitext(file_name)
                if extension == '.pdf':
                    loader = PyPDFLoader(file_name)
                elif extension == '.docx':
                    loader = Docx2txtLoader(file_name)
                elif extension == '.txt':
                    loader = TextLoader(file_name)
                else:
                    st.error('Document format is not supported!')
                    return
                document = loader.load()
                initialize_document(document)

    else:
        st.error('No file uploaded')
        return



    print('se cargo el documento')
def split_in_chunks(document: List[str], chunk_size: int = 1000, chunk_overlap: int = 200) -> List[str]:
    '''
    SPLIT DOCUMENT IN CHUNKS
    :param document: Document to be split
    :param chunk_size: Size of chunks
    :param chunk_overlap: Overlap of chunks
    :return: chunked documents
    '''
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", " ", ""]
    )

    document_chunks = text_splitter.split_documents(document)
    return document_chunks

def generate_embeddings(document: List[str]) -> OpenAIEmbeddings:
    '''
    GENERATE OpenAI EMBEDDINGS
    '''
    embeddings = OpenAIEmbeddings(openai_api_key=API_KEY)
    return embeddings

def generate_vector_retrieval(chunks: List[List[str]], embeddings: OpenAIEmbeddings) -> Chroma:
    '''
    CREATE VECTOR SYSTEM RETRIEVAL
    :param chunck: chunked documents
    :param embeddings: OpenAI EMBEDDINGS
    :return: Chroma vector retriever
    '''
    chromadb = Chroma.from_documents(chunks, embeddings)
    vector_retrieval = chromadb.as_retriever()
    return vector_retrieval

def generate_llm(api_key: str, model_name: str, temperature: float) -> ChatOpenAI:
    '''
    CREATE  LLM
    :param api_key: OpenAI API key
    :param model_name: Model name
    :param temperature: Model temperature
    :return: OpenAI LM
    '''
    llm_chat = ChatOpenAI(
        model_name=model_name,
        temperature=temperature,
        openai_api_key=api_key
    )

    return llm_chat

def create_chat(llm_chat: ChatOpenAI , retrieval_chain: Chroma)->ConversationalRetrievalChain:
    '''
    CREATE CONVERSATIONAL RETRIEVAL CHAIN
    :param llm_chat: OpenAI LM
    :param retrieval_chain: OpenAI RETRIEVAL CHAIN
    :return: Conversational retrieval chain
    '''
    crc_chain = ConversationalRetrievalChain.from_llm(
        llm=llm_chat,
        retriever=retrieval_chain
    )
    crc_chain = crc_chain
    return crc_chain

def initialize_document(document):
    clear_history()
    chunks = split_in_chunks(document)
    embeddings = generate_embeddings(chunks)
    vector_retrieval = generate_vector_retrieval(chunks, embeddings)
    llm = generate_llm(API_KEY, MODEL_LLM, MODEL_TEMPERATURE)
    crc_chain = create_chat(llm, vector_retrieval)
    st.session_state.crc = crc_chain
    st.success('File uploaded, chunked and embedded successfully')

###########################################
#CARGAR VARIABLES DE ENTORNO
load_dotenv(PATH_ENV_VARIABLES)
API_KEY = os.getenv(API_KEY_ENV)
if not API_KEY:
    print('Did not find API key')
    raise ValueError('Did not find API key from OpenAI')


###########################################
#GENERACION DEL UI
st.title('Chat with Document')

#Carga y elimina el historial del documento anterior.
uploaded_file = st.file_uploader('Upload file', type=['pdf', 'docx', 'txt'])
add_file = st.button('Add File', on_click=uploaded_text)

###########################################
#CARGA Y CONFIGURACION DE DOCUMENTOS



query = st.text_input('Input yor question')

###########################################
#VALIDACION Y RESPUESTA A QUERY
if query:
    #Get the crc from Session
    if 'crc' in st.session_state:
        #Run the query with context
        response = st.session_state.crc.run({
            'question': query,
            'chat_history': st.session_state['history']
        })

    #Update context with a tuple query and response
    st.session_state['history'].append((query, response))
    #Show answer
    st.subheader('Answer')
    st.write(response)
    #Mostramos el historial:
    #st.write(st.session_state['history'])
    for prompts in st.session_state['history']:
        st.write("Question: " + prompts[0])
        st.write("Answer: " + prompts[1])
