"""Punto de entrada: chat por consola con el asistente de viajes.

Uso:
    python main.py                                        # modo conversación
    python main.py "¿Qué barrio de Lisboa me recomiendas?"   # una sola pregunta
"""

import asyncio
import sys

from langchain_core.messages import HumanMessage

from app.agente import construir_agente


async def preguntar(agente, historial: list, pregunta: str) -> None:
    """Envía una pregunta al agente y muestra qué herramientas va usando."""
    historial.append(HumanMessage(pregunta))

    # Las herramientas MCP son asíncronas, por eso usamos astream en vez de stream
    async for paso in agente.astream({"messages": historial}, stream_mode="updates"):
        for cambios in paso.values():
            for mensaje in (cambios or {}).get("messages", []):
                historial.append(mensaje)  # guardamos todo para tener memoria
                if mensaje.type == "ai" and mensaje.tool_calls:
                    for llamada in mensaje.tool_calls:
                        print(f"  [herramienta] {llamada['name']}({llamada['args']})")
                elif mensaje.type == "ai":
                    print(f"\nAsistente: {mensaje.text}\n")


async def main() -> None:
    print("Preparando el agente (servidor MCP de Airbnb e índice de documentos)...")
    agente = await construir_agente()
    historial = []

    # Si se pasa una pregunta como argumento, se responde y se termina
    if len(sys.argv) > 1:
        await preguntar(agente, historial, " ".join(sys.argv[1:]))
        return

    print("Asistente de Rumbo Sur Viajes. Escribe 'salir' para terminar.\n")
    while True:
        pregunta = input("Tú: ").strip()
        if pregunta.lower() in {"salir", "exit", "quit"}:
            break
        if pregunta:
            await preguntar(agente, historial, pregunta)


if __name__ == "__main__":
    asyncio.run(main())
