from transformers import AutoModelForSequenceClassification, AutoTokenizer
import os

# Nombre del modelo en Hugging Face
model_name = "cardiffnlp/twitter-xlm-roberta-base-sentiment"

# Descarga el modelo y el tokenizer
model = AutoModelForSequenceClassification.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)

# Crea el directorio para guardar el modelo
os.makedirs(model_name, exist_ok=True)

# Guarda el modelo y tokenizer localmente
model.save_pretrained(model_name)
tokenizer.save_pretrained(model_name)