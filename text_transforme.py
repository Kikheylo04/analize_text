from transformers import pipeline

# Cargar un modelo más robusto para análisis de sentimiento
analizador = pipeline("sentiment-analysis", model="siebert/sentiment-roberta-large-english")

comentario = "Estuvo bien crudo"
resultado = analizador(comentario)[0]

# Mostrar el resultado
print(f"Comentario: {comentario}")
print(f"Sentimiento: {resultado['label']}, Puntaje: {resultado['score']}")
