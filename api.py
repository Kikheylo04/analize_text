import boto3
import json

# Cliente de SageMaker para hacer la inferencia
client = boto3.client('sagemaker-runtime', region_name='us-east-2',aws_access_key_id="",aws_secret_access_key="")

# Definir el texto de entrada para la inferencia
text = "Estuvo bien crudo"

input_data = {
    "inputs": text
}

# Enviar la solicitud de inferencia al endpoint
response = client.invoke_endpoint(
    EndpointName="huggingface-pytorch-inference-2024-09-20-17-08-07-112",  # Nombre de tu endpoint
    ContentType='application/json',  # Tipo de contenido
    Body=json.dumps(input_data)  # Datos de entrada en formato JSON
)

# Leer la respuesta
result = json.loads(response['Body'].read().decode())

print(f"Texto: {text} ------- Resultado: {result}")