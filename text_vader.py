from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from comentario.comentario import comentarios

def clasificar_comentario_vader(comentario):
    analyzer = SentimentIntensityAnalyzer()
    puntaje = analyzer.polarity_scores(comentario)
    
    if puntaje['compound'] >= 0.05:
        return "Positivo"
    elif puntaje['compound'] <= -0.05:
        return "Negativo"
    else:
        return "Neutral"

for c in comentarios:
    print(f"Comentario: {c} - Clasificación: {clasificar_comentario_vader(c)}")
