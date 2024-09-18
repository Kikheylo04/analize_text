from textblob import TextBlob
from comentario.comentario import comentarios

def clasificar_comentario(comentario):
    analisis = TextBlob(comentario)
    # Analiza el sentimiento
    if analisis.sentiment.polarity > 0:
        return "Positivo"
    elif analisis.sentiment.polarity < 0:
        return "Negativo"
    else:
        return "Neutral"

for c in comentarios:
    print(f"Comentario: {c} - Clasificación: {clasificar_comentario(c)}")
