"""Tests del buscador híbrido sobre los documentos de docs/."""

from app.rag import CARPETA_DOCS, BuscadorHibrido, cargar_fragmentos, crear_herramienta_rag


def test_los_documentos_se_trocean_por_secciones():
    fragmentos = cargar_fragmentos(CARPETA_DOCS)

    assert len(fragmentos) > 15
    # Cada fragmento lleva delante el título de su documento
    assert any(f["texto"].startswith("Guía de Lisboa. Barrios recomendados") for f in fragmentos)


def test_un_codigo_exacto_encuentra_su_seccion(embeddings_falsos):
    buscador = BuscadorHibrido(embeddings=embeddings_falsos)

    mejor = buscador.buscar("¿Qué incluye la tarifa RS-ECO?")[0]

    assert mejor["fuente"] == "politica_cancelacion.md"
    assert "RS-ECO" in mejor["texto"]


def test_una_pregunta_sobre_un_destino_encuentra_su_guia(embeddings_falsos):
    buscador = BuscadorHibrido(embeddings=embeddings_falsos)

    fuentes = [r["fuente"] for r in buscador.buscar("barrios recomendados en Oporto")]

    assert fuentes[0] == "guia_oporto.md"


def test_la_herramienta_devuelve_texto_con_la_fuente(embeddings_falsos):
    herramienta = crear_herramienta_rag(BuscadorHibrido(embeddings=embeddings_falsos))

    texto = herramienta.invoke({"consulta": "¿puedo viajar con mascotas?"})

    assert "[Fuente: preguntas_frecuentes.md]" in texto
