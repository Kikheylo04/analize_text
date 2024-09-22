import sagemaker
import json

# Configura el predictor
predictor = sagemaker.predictor.Predictor(endpoint_name="huggingface-pytorch-inference-2024-09-20-14-53-42-948")

# Definir el texto de entrada para la inferencia
input_data = {
    "inputs": "Este es un gran día para el análisis de sentimientos."
}

# Realizar la predicción
response = predictor.predict(json.dumps(input_data))

# Mostrar el resultado
print(response)