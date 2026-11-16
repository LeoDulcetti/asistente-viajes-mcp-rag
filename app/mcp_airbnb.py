"""Conexión con el servidor MCP público de Airbnb.

MCP (Model Context Protocol) es un estándar para ofrecer herramientas a un modelo.
El servidor es un programa aparte (aquí, un paquete de Node que se lanza con npx):
nosotros solo nos conectamos y le pedimos su lista de herramientas.
No hace falta cuenta ni clave de Airbnb.
"""

import os

from langchain_mcp_adapters.client import MultiServerMCPClient


async def cargar_herramientas_airbnb() -> list:
    """Arranca el servidor MCP de Airbnb y devuelve sus herramientas ya listas para LangChain.

    Herramientas que ofrece: airbnb_search y airbnb_listing_details.
    """
    argumentos = ["-y", "@openbnb/mcp-server-airbnb"]

    # Por defecto el servidor respeta el robots.txt de Airbnb (ver README)
    if os.getenv("AIRBNB_IGNORAR_ROBOTS", "false").lower() == "true":
        argumentos.append("--ignore-robots-txt")

    cliente = MultiServerMCPClient(
        {
            "airbnb": {
                "transport": "stdio",  # el servidor es un proceso local; hablamos por entrada/salida
                "command": "npx",
                "args": argumentos,
            }
        }
    )
    return await cliente.get_tools()
