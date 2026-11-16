"""RAG sencillo con búsqueda híbrida sobre la carpeta docs/.

La búsqueda híbrida combina dos buscadores que se complementan:

- BM25 (palabras clave): acierta cuando la pregunta usa las mismas palabras que el
  documento. Es imbatible con códigos y nombres propios ("RS-FLEX", "Triana").
- Vectorial (significado): acierta aunque se usen palabras distintas
  ("anular la reserva" encuentra "política de cancelación").

Cada buscador devuelve su ranking y los fusionamos con Reciprocal Rank Fusion (RRF).
"""

import re
from pathlib import Path

import numpy as np
from langchain_core.tools import tool
from rank_bm25 import BM25Okapi

CARPETA_DOCS = Path(__file__).parent.parent / "docs"

# Modelo de embeddings multilingüe que se ejecuta en local (gratis, sin clave de API)
MODELO_EMBEDDINGS = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"


class EmbeddingsLocales:
    """Convierte textos en vectores con un modelo local de la librería fastembed."""

    def __init__(self):
        from fastembed import TextEmbedding

        # La primera vez descarga el modelo (unos 220 MB) y lo guarda en caché
        self.modelo = TextEmbedding(MODELO_EMBEDDINGS)

    def embed(self, textos: list[str]) -> np.ndarray:
        return np.array(list(self.modelo.embed(textos)))


def cargar_fragmentos(carpeta: Path) -> list[dict]:
    """Lee los .md de la carpeta y los trocea por secciones (cada título '## ').

    Trocear por secciones mantiene junta la información de un mismo tema.
    """
    fragmentos = []
    for fichero in sorted(carpeta.glob("*.md")):
        texto = fichero.read_text(encoding="utf-8")
        titulo, *secciones = texto.split("\n## ")
        titulo = titulo.strip().lstrip("# ")
        for seccion in secciones:
            # Añadimos el título del documento para que el fragmento se entienda solo
            fragmentos.append({"fuente": fichero.name, "texto": f"{titulo}. {seccion.strip()}"})
    return fragmentos


def tokenizar(texto: str) -> list[str]:
    """Divide un texto en palabras en minúsculas (lo que necesita BM25)."""
    return re.findall(r"\w+", texto.lower())


class BuscadorHibrido:
    """Indexa los fragmentos una vez y responde búsquedas híbridas."""

    def __init__(self, carpeta: Path = CARPETA_DOCS, embeddings=None):
        self.fragmentos = cargar_fragmentos(carpeta)
        textos = [f["texto"] for f in self.fragmentos]

        # Índice 1: BM25 sobre las palabras de cada fragmento
        self.bm25 = BM25Okapi([tokenizar(t) for t in textos])

        # Índice 2: un vector por fragmento (una matriz de numpy hace de "base vectorial")
        self.embeddings = embeddings or EmbeddingsLocales()
        self.vectores = self._normalizar(self.embeddings.embed(textos))

    @staticmethod
    def _normalizar(vectores: np.ndarray) -> np.ndarray:
        """Deja los vectores con longitud 1: así el producto escalar es la similitud coseno."""
        return vectores / np.linalg.norm(vectores, axis=1, keepdims=True)

    def _ranking_bm25(self, consulta: str) -> list[int]:
        """Índices de los fragmentos ordenados por coincidencia de palabras."""
        puntuaciones = self.bm25.get_scores(tokenizar(consulta))
        orden = np.argsort(puntuaciones)[::-1]
        # Descartamos los fragmentos que no comparten ninguna palabra con la consulta
        return [int(i) for i in orden if puntuaciones[i] > 0]

    def _ranking_vectorial(self, consulta: str) -> list[int]:
        """Índices de los fragmentos ordenados por parecido de significado."""
        vector = self._normalizar(self.embeddings.embed([consulta]))[0]
        similitudes = self.vectores @ vector
        return [int(i) for i in np.argsort(similitudes)[::-1]]

    def buscar(self, consulta: str, k: int = 3) -> list[dict]:
        """Devuelve los k fragmentos más relevantes fusionando los dos rankings."""
        puntos: dict[int, float] = {}
        for ranking in (self._ranking_bm25(consulta), self._ranking_vectorial(consulta)):
            for posicion, indice in enumerate(ranking[:10]):
                # RRF: cada buscador da más puntos cuanto más arriba coloca el fragmento.
                # El 60 es la constante habitual; suaviza la diferencia entre posiciones.
                puntos[indice] = puntos.get(indice, 0) + 1 / (60 + posicion + 1)

        mejores = sorted(puntos, key=puntos.get, reverse=True)[:k]
        return [self.fragmentos[i] for i in mejores]


def crear_herramienta_rag(buscador: BuscadorHibrido):
    """Envuelve el buscador en una herramienta que el agente puede llamar."""

    @tool
    def buscar_en_guias(consulta: str) -> str:
        """Busca en la documentación interna de Rumbo Sur Viajes: guías de Lisboa, Oporto
        y Sevilla (barrios, transporte, mejor época), política de cancelación, seguro de
        viaje y preguntas frecuentes."""
        resultados = buscador.buscar(consulta)
        return "\n\n".join(f"[Fuente: {r['fuente']}]\n{r['texto']}" for r in resultados)

    return buscar_en_guias
