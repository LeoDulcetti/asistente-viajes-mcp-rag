"""Piezas falsas para probar el proyecto sin descargar modelos ni usar claves de API."""

import zlib

import numpy as np
import pytest

from app.rag import tokenizar


class EmbeddingsFalsos:
    """Embeddings de pega: cada palabra suma 1 en una posición fija del vector.

    No entienden el significado (solo coinciden palabras), pero tienen la misma
    interfaz que EmbeddingsLocales y bastan para probar el mecanismo.
    """

    def embed(self, textos: list[str]) -> np.ndarray:
        vectores = np.full((len(textos), 256), 1e-6)
        for fila, texto in enumerate(textos):
            for palabra in tokenizar(texto):
                vectores[fila, zlib.crc32(palabra.encode()) % 256] += 1
        return vectores


@pytest.fixture
def embeddings_falsos():
    return EmbeddingsFalsos()
