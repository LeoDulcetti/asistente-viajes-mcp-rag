"""Configuración: carga el fichero .env y crea el modelo de lenguaje."""

import os

from dotenv import load_dotenv
from langchain.chat_models import init_chat_model

# Lee las variables del fichero .env (por ejemplo, GROQ_API_KEY)
load_dotenv()


def crear_modelo():
    """Crea el modelo de chat del agente.

    Por defecto usa un modelo abierto servido por Groq, que tiene un plan gratuito.
    El formato es "proveedor:modelo"; se cambia con la variable MODELO.
    """
    return init_chat_model(os.getenv("MODELO", "groq:llama-3.3-70b-versatile"))
