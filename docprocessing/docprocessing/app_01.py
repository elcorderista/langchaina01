import os
import streamlit as st
from dotenv import load_dotenv
from langchain.prompts import PromptTemplate
from langchain_openai import OpenAI
from langchain.schema.runnable import RunnableSequence


#Importacion de chains
from langchain.chains import LLMChain as Chain



load_dotenv('../.env')
API_KEY = os.getenv('OPENAIKEY')

#Configuramos st
st.title('Medium Article Generator')
topic = st.text_input('Input your topic of interest')
list_languages = ['Spanish', 'English', 'Portuguese', 'German', 'Russian', 'Hindi']
language = st.selectbox('Select language', list_languages)


#Generamos un template con una matriz de variables y la respuesta
title_template = PromptTemplate(
    input_variables=['topic', 'language'],
    template='Give me a medium article title on {topic} in {language}'
)

#Instanciamos openaAi con una temperatura alta
llm = OpenAI(temperature=0.9, openai_api_key=API_KEY)

title_chain = RunnableSequence(title_template, llm)

if topic:
    try:
        response = title_chain.invoke({'topic': topic, 'language': language, 'verbose': True})
        # Mostrar la respuesta
        st.subheader('Generated Article Title')
        st.write(response)
    except Exception as e:
        st.error(f'Error al generar el texto: {e}')
