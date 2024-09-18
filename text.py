from transformers import pipeline
from comentario.comentario import comentarios

classifier = pipeline("text-classification", model="nlptown/bert-base-multilingual-uncased-sentiment")

resultados = classifier(comentarios)

for i, resultado in enumerate(resultados):
    print(f"Comentario {i+1}: {comentarios[i]}")
    print(f"Etiqueta: {resultado['label']}, Puntaje: {resultado['score']}\n")
