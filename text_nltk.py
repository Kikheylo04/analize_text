from nltk.sentiment.vader import SentimentIntensityAnalyzer
import nltk
from comentario.comentario import comentarios

# Descargar el lexicón de VADER
nltk.download('vader_lexicon')

# Crear una instancia del analizador de sentimiento
sid = SentimentIntensityAnalyzer()

for comentario in comentarios:
    resultado = sid.polarity_scores(comentario)
    print(f"Análisis de Sentimiento: {resultado}")
