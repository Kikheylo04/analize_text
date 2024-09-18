import spacy
from textblob import TextBlob

# Cargar el modelo de spaCy
nlp = spacy.load("en_core_web_sm")

# Texto para análisis
texto = "Estuvo bien crudo"

# Procesar el texto con spaCy
doc = nlp(texto)

# Usar TextBlob para análisis de sentimientos
blob = TextBlob(doc.text)

# Mostrar el análisis de sentimiento
print(f"Sentimiento: {blob.sentiment}")
