"""Test del agente completo con un modelo falso y una herramienta MCP simulada."""

import asyncio

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from langchain_core.tools import tool

from app.agente import MAX_CARACTERES, construir_agente


class ModeloFalso(BaseChatModel):
    """Modelo de pega: devuelve, en orden, las respuestas que le preparamos."""

    respuestas: list

    @property
    def _llm_type(self) -> str:
        return "modelo-falso"

    def _generate(self, messages, stop=None, run_manager=None, **kwargs):
        return ChatResult(generations=[ChatGeneration(message=self.respuestas.pop(0))])

    def bind_tools(self, tools, **kwargs):
        return self


@tool
async def airbnb_search(location: str) -> str:
    """Herramienta que imita a la del servidor MCP y devuelve un texto larguísimo."""
    return f"Alojamientos en {location}: " + "x" * 50_000


def test_el_agente_combina_rag_y_mcp(embeddings_falsos):
    modelo = ModeloFalso(
        respuestas=[
            # El modelo pide usar las dos herramientas a la vez
            AIMessage(
                "",
                tool_calls=[
                    {"name": "buscar_en_guias", "args": {"consulta": "barrios de Lisboa"}, "id": "1"},
                    {"name": "airbnb_search", "args": {"location": "Lisboa"}, "id": "2"},
                ],
            ),
            AIMessage("Te recomiendo Baixa. Aquí tienes alojamientos."),
        ]
    )

    async def ejecutar():
        agente = await construir_agente(modelo, embeddings_falsos, herramientas_mcp=[airbnb_search])
        return await agente.ainvoke({"messages": [HumanMessage("Busca alojamiento en un buen barrio de Lisboa")]})

    mensajes = asyncio.run(ejecutar())["messages"]
    resultados = {m.name: m.text for m in mensajes if m.type == "tool"}

    # El RAG ha devuelto la guía correcta
    assert "guia_lisboa.md" in resultados["buscar_en_guias"]
    # El resultado enorme de Airbnb se ha recortado antes de llegar al modelo
    assert len(resultados["airbnb_search"]) < MAX_CARACTERES + 100
    assert mensajes[-1].text == "Te recomiendo Baixa. Aquí tienes alojamientos."
