from textblob import TextBlob

def clasificar_comentario(comentario):
    analisis = TextBlob(comentario)
    if analisis.sentiment.polarity > 0:
        return "Positivo"
    elif analisis.sentiment.polarity < 0:
        return "Negativo"
    else:
        return "Neutral"

print(f"Clasificación: {clasificar_comentario('Estuvo feo')}")
