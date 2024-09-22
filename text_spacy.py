import spacy
from textblob import TextBlob

# Cargar el modelo de spaCy
nlp = spacy.load("es_core_news_md")

# Texto para análisis
text = "Estuvo rico"

# Procesar el texto con spaCy
doc = nlp(text)

# Usar TextBlob para análisis de sentimientos
blob_with_spacy = TextBlob(doc.text)
blob_without_spacy = TextBlob(text)

# Mostrar el análisis de sentimiento
print(f"Sentimiento con spacy: {blob_with_spacy.sentiment}")
print(f"Sentimiento sin spacy: {blob_without_spacy.sentiment}")
