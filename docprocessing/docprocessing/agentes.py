import os
import streamlit as st
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain.agents import Tool, initialize_agent, AgentType
from langchain_community.tools import WikipediaQueryRun
from langchain_community.utilities import WikipediaAPIWrapper
from langchain_experimental.tools import PythonREPLTool

# Carga de variables de entorno
load_dotenv('../.env')
API_KEY = os.getenv('OPENAIKEY')

# Configurar Streamlit
st.title('Agent Using Wikipedia and Calculator')
topic = st.text_input("Input your topic of interest")

# Configurar modelo LLM (Modelo de chat)
llm = ChatOpenAI(model_name="gpt-4", temperature=0.0, openai_api_key=API_KEY)

# Configurar herramientas personalizadas
tools = [
    Tool(
        name="Wikipedia",
        func=WikipediaQueryRun(api_wrapper=WikipediaAPIWrapper()).run,
        description="Busca información en Wikipedia sobre cualquier tema. Ideal para fechas y eventos."
    ),
    Tool(
        name="Calculator",
        func=PythonREPLTool().run,
        description="Realiza cálculos matemáticos complejos."
    ),
]

# Crear el agente
agent = initialize_agent(
    tools=tools,
    llm=llm,
    agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
    verbose=True
)

# Ejecutar el agente si se proporciona un tema
if topic:
    try:
        response = agent.run(topic)
        st.write(response)
    except Exception as e:
        st.error(f"Error: {e}")
