#Importacion de librerias

import os
import streamlit as st                                                                  #Crear la interfaz grafica
from dotenv import load_dotenv                                                          #Leer variables de entorno
from langchain_openai import ChatOpenAI                                                 #Importar el modelo
from langchain_openai import OpenAIEmbeddings                                           #Creacion de embeddings
from langchain_community.document_loaders import TextLoader                             #Cargar documentos
from langchain_community.vectorstores import Chroma                                     #Generar la base de datos
from langchain.text_splitter import RecursiveCharacterTextSplitter                      #Generar los chuncks
from langchain.chains.conversational_retrieval.base import ConversationalRetrievalChain #Chat con memoria
from typing import List                                                                 #Documentacion de codigo

###########################################
#VARIABLES GLOBALES
PATH_DOCUMENT =  './data/us-constitution.txt'
MODEL_LLM = "gpt-3.5-turbo"
PATH_ENV_VARIABLES = '../.env'
API_KEY_ENV = 'OPENAIKEY'
MODEL_TEMPERATURE = 0.0

###########################################
#CARGAR VARIABLES DE ENTORNO
load_dotenv(PATH_ENV_VARIABLES)
API_KEY = os.getenv(API_KEY_ENV)
if not API_KEY:
    print('Did not find API key')
    raise ValueError('Did not find API key from OpenAI')

###########################################
#CARGAR DE DOCUMENTO
loader = TextLoader(PATH_DOCUMENT)
document = loader.load()

###########################################
#GENERACION DE CHUNKS
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size= 1000,
    chunk_overlap= 200,
    separators=["\n\n", "\n", " ", ""]
)

document_chunks = text_splitter.split_documents(document)

###########################################
#GENERACION DEL MODELO
llm_chat = ChatOpenAI(
    model_name=MODEL_LLM,
    temperature=MODEL_TEMPERATURE,
    openai_api_key=API_KEY
)

###########################################
#GENERACION Y ALMACENAMIENTO DE EMBEDDINGS
embeddings = OpenAIEmbeddings(openai_api_key=API_KEY)
vectorStore_chromaDB = Chroma.from_documents(document_chunks,embeddings)
retriever = vectorStore_chromaDB.as_retriever()

###########################################
#GENERACION DEL CHAT
crc_chain = ConversationalRetrievalChain.from_llm(
    llm=llm_chat,
    retriever=retriever
)

###########################################
#GENERACION DEL UI
st.title('Chat with Document')

query = st.text_input('Input yor question')

###########################################
#VALIDACION Y RESPUESTA A QUERY
if query:
    #Manejamos la sesion del chat
    if 'history' not in st.session_state:
        st.session_state['history'] = []

    #Run the query with context
    response = crc_chain.run({
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
