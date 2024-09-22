from transformers import pipeline, AutoTokenizer
import warnings

warnings.filterwarnings("ignore", category=FutureWarning)
tokenizer = AutoTokenizer.from_pretrained("cardiffnlp/twitter-xlm-roberta-base-sentiment", use_fast=False)

analizador = pipeline("sentiment-analysis", model="cardiffnlp/twitter-xlm-roberta-base-sentiment", device=0, tokenizer=tokenizer)

text = "sabor"

resultado = analizador(text)

print(f"Texto: {text}", "      ",resultado)



# !pip install transformers torch sentencepiece 

# tokenizer = AutoTokenizer.from_pretrained("cardiffnlp/twitter-xlm-roberta-base-sentiment", use_fast=False)
# sentiment_analyzer = pipeline("sentiment-analysis", model="cardiffnlp/twitter-xlm-roberta-base-sentiment", device=-1, tokenizer=tokenizer)
# result = sentiment_analyzer(text)





import sagemaker
from sagemaker.model import Model

role = 'arn:aws:iam::530166493863:role/service-role/AmazonSageMaker-ExecutionRole-20240919T171030'

# Crea el modelo en SageMaker usando tu script personalizado
model = Model(
    entry_point='app.py',  # El archivo que tiene tu `model_fn` y `predict_fn`
    role=role,
    image_uri='763104351884.dkr.ecr.us-east-2.amazonaws.com/pytorch-inference:2.0.0-gpu-py310',  # Imagen para la región us-east-2
)

# Despliega el modelo como un endpoint
predictor = model.deploy(
    initial_instance_count=1,
    instance_type='ml.t2.medium'  # Instancia pequeña para pruebas
)

print("Endpoint creado con éxito.")