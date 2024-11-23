import os
import streamlit as st
from dotenv import load_dotenv
from langchain.llms import OpenAI

#Importacion de templates
from langchain.prompts import PromptTemplate

load_dotenv('../.env')
API_KEY = os.getenv('OPENAIKEY')

st.title('Medium Article Generator')
topic = st.text_input('Input your topic of interest')

#Generamos un template con una matriz de variables y la respuesta
title_template = PromptTemplate(
    input_variables=['topic', 'language'],
    template='Give me a medium article title on {topic} in {language}'
)

#Instanciamos openaAi con una temperatura alta
llm = OpenAI(temperature=0.9, openai_api_key=API_KEY)

if topic:
    try:
        #Generamos el promt usando el template
        #prompt = title_template.format(topic=topic, language='en')
        prompt = title_template.format(topic=topic, language='Spanish')
        #Establecemos la respuesta en un lenguaje especifico
        response = llm(prompt)

        #Mostrar la respuesta
        st.subheader('Generated Article Title')
        st.write(response)
    except Exception as e:
        st.error(f'Error al generar el texto: {e}')


