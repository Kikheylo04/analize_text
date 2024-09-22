from transformers import AutoModelForSequenceClassification, AutoTokenizer
import torch

# Cargar el modelo y tokenizer entrenado desde la carpeta
model = AutoModelForSequenceClassification.from_pretrained("./modelo_urgencia")
tokenizer = AutoTokenizer.from_pretrained("./modelo_urgencia")

# Definir el mapeo inverso de las etiquetas
label_mapping_inverse = {0: "Alta", 1: "Moderada", 2: "Baja"}

# Función para hacer predicciones
def predecir_urgencia(texto):
    # Tokenizar el texto de entrada
    inputs = tokenizer(texto, return_tensors="pt", truncation=True, padding=True, max_length=128)
    
    # Desactivar el cálculo de gradientes (no es necesario para la inferencia)
    with torch.no_grad():
        # Pasar los inputs al modelo
        outputs = model(**inputs)
    
    # Obtener las probabilidades (logits)
    logits = outputs.logits
    
    # Obtener la predicción como la clase con mayor probabilidad
    prediccion = torch.argmax(logits, dim=1).item()

    print("prediccon", prediccion)
    
    # Convertir la predicción en una etiqueta legible
    return label_mapping_inverse[prediccion]

# Ejemplo de uso
texto_ejemplo = "Estuvo bien crudo"
resultado = predecir_urgencia(texto_ejemplo)
print(f"La urgencia del texto es: {resultado}")
