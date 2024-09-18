from transformers import pipeline
from comentario.comentario import comentarios

# Cargar el pipeline de análisis de sentimientos con el modelo multilingüe
sentiment_pipeline = pipeline("sentiment-analysis", model="nlptown/bert-base-multilingual-uncased-sentiment")

# Función para mapear las estrellas a categorías sentimentales
def clasificar_sentimiento(estrellas):
    if estrellas in [1, 2]:
        return "NEGATIVO"
    elif estrellas == 3:
        return "NEUTRAL"
    else:
        return "POSITIVO"

def clasificar_confianza(resultado):
    if estrellas == "NEGATIVO":
        return resultado
    elif estrellas == "NEUTRAL":
        return resultado
    else:
        return resultado

# Clasificar los comentarios
resultados = sentiment_pipeline(comentarios)

# Mostrar los resultados con la clasificación de sentimiento personalizada
for comentario, resultado in zip(comentarios, resultados):
    estrellas = int(resultado['label'].split()[0])  # Obtener la cantidad de estrellas (1 a 5)
    sentimiento = clasificar_sentimiento(estrellas)
    clasificacion = clasificar_confianza(estrellas)
    print(f"Comentario: {comentario}")
    print(f"Sentimiento: {sentimiento}, Calidad: {clasificacion}\n")
  
