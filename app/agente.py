"""El agente: un modelo con dos tipos de herramientas.

- Herramientas MCP de Airbnb -> datos en vivo de alojamientos.
- Herramienta RAG            -> documentación interna de la agencia.

El modelo decide en cada pregunta cuál usar (o las dos).
"""

from langchain.agents import create_agent
from langchain.agents.middleware import wrap_tool_call

from app.config import crear_modelo
from app.mcp_airbnb import cargar_herramientas_airbnb
from app.rag import BuscadorHibrido, crear_herramienta_rag

# Máximo de caracteres del resultado de una herramienta que se envía al modelo
MAX_CARACTERES = 6000

PROMPT = """Eres el asistente de Rumbo Sur Viajes, una agencia de viajes.

- Para preguntas sobre destinos (barrios, transporte, mejor época), cancelaciones, seguro
  o normas de la agencia, usa la herramienta buscar_en_guias y cita el fichero de la fuente.
- Para buscar alojamientos reales, usa las herramientas de Airbnb. Las fechas van en
  formato AAAA-MM-DD. Muestra como mucho 3 alojamientos, con precio y enlace.
- Si una pregunta necesita las dos cosas, usa las dos.
- No inventes datos: si las herramientas no devuelven la información, dilo."""


@wrap_tool_call
async def recortar_resultado(peticion, ejecutar):
    """Recorta los resultados muy largos (Airbnb devuelve mucho texto).

    Así no superamos el límite de tokens por minuto de los modelos gratuitos.
    """
    resultado = await ejecutar(peticion)
    texto = resultado.text
    if len(texto) > MAX_CARACTERES:
        resultado.content = texto[:MAX_CARACTERES] + "\n[... resultado recortado]"
    return resultado


async def construir_agente(modelo=None, embeddings=None, herramientas_mcp=None):
    """Crea el agente. Los parámetros permiten sustituir piezas en los tests."""
    if herramientas_mcp is None:
        herramientas_mcp = await cargar_herramientas_airbnb()

    herramienta_rag = crear_herramienta_rag(BuscadorHibrido(embeddings=embeddings))

    return create_agent(
        modelo or crear_modelo(),
        tools=[herramienta_rag, *herramientas_mcp],
        system_prompt=PROMPT,
        middleware=[recortar_resultado],
    )
