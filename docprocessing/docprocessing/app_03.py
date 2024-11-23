import os
import streamlit as st
from dotenv import load_dotenv
from langchain.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
from langchain.schema.runnable import RunnableSequence

#Carga de variables de entorno
load_dotenv('../.env')
API_KEY = os.getenv('OPENAIKEY')

#Configuramos st
st.title('Medium Article Generator')
topic = st.text_input('Input your topic of interest')
list_languages = ['Spanish', 'English', 'Portuguese', 'German', 'Russian', 'Hindi']
language = st.selectbox('Select language', list_languages)

#Configuracion de Templates

#Step 1: Generate the title:
title_template = PromptTemplate(
    input_variables=['topic', 'language'],
    template='Give me a medium article title on {topic} in {language}'
)
#Step 2: Generate summary
article_template = PromptTemplate(
    input_variables=['title'],
    template=(
        "Write a short summary for a Medium article titled: {title}. "
        "The summary should be concise and no longer than 100 words.")
)

#Configuracion de modelos
llm_title = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0.7, openai_api_key=API_KEY)
llm_summary = ChatOpenAI(model_name="gpt-4", temperature=0.9, openai_api_key=API_KEY)

#CREAMOS LAS SECUIENCIAS DE PROCESAMIENTOI
title_chain = RunnableSequence(title_template, llm_title)
summary_chain = RunnableSequence(article_template, llm_summary)


#Generacion de flujo cuando hay un tema
if topic:
    try:
        # Step 1:
        generated_title = title_chain.invoke({'topic': topic, 'language': language})
        # Step 2:
        article_content = summary_chain.invoke({'title': generated_title, 'language': language})
        # Mostrar resultados
        st.subheader('Generated Article Title con GPT 3.5 Turbo:')
        st.write(generated_title.content)

        st.subheader('Generated Article Content con GPT 4:')
        st.write(article_content.content)

    except Exception as e:
        st.error(f'Error en la generacion de Chains: {e}')