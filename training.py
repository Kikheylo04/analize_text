from transformers import AutoModelForSequenceClassification, Trainer, TrainingArguments, AutoTokenizer
from datasets import load_dataset
import torch
import warnings

warnings.filterwarnings("ignore", category=FutureWarning)

# Cargar el tokenizer y el modelo DistilBERT preentrenado
tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")  # DistilBERT es mucho más ligero
model = AutoModelForSequenceClassification.from_pretrained("distilbert-base-uncased", num_labels=3)

# Cargar el dataset desde CSV
dataset = load_dataset('csv', data_files={'train': 'datos_urgencia_train.csv', 'test': 'datos_urgencia_test.csv'})

# Mapear las etiquetas de texto a números
label_mapping = {"Alta": 0, "Moderada": 1, "Baja": 2}

def encode_labels(example):
    example['labels'] = label_mapping[example['urgency']]  # El campo de etiquetas debe llamarse 'labels'
    return example

dataset = dataset.map(encode_labels)

# Tokenizar los textos con truncamiento y padding
max_length = 128  # Ajusta la longitud máxima según lo que necesites
dataset = dataset.map(lambda examples: tokenizer(examples['text'], truncation=True, padding='max_length', max_length=max_length), batched=True)

# Definir los argumentos del entrenamiento con optimización de memoria
training_args = TrainingArguments(
    output_dir="./results",  # Carpeta de salida
    evaluation_strategy="epoch",  # Evaluar después de cada época
    learning_rate=2e-5,  # Tasa de aprendizaje
    per_device_train_batch_size=16,  # Puedes probar con un tamaño de lote más grande ya que DistilBERT es más ligero
    per_device_eval_batch_size=32,  # Batch size más grande en evaluación
    num_train_epochs=3,  # Número de épocas
    weight_decay=0.01,  # Decaimiento de pesos para evitar sobreajuste
    gradient_accumulation_steps=2,  # Acumula gradientes cada 2 pasos para simular un batch más grande
    fp16=True,  # Usa precisión flotante de 16 bits (si la GPU lo soporta)
    no_cuda=False,  # Asegúrate de que estamos utilizando la GPU (puedes cambiar a True para usar CPU)
    logging_dir='./logs',  # Directorio para los logs
)

# Crear el objeto Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=dataset['train'],
    eval_dataset=dataset['test'],
)

# Entrenar el modelo
trainer.train()

# Liberar memoria de la GPU después del entrenamiento
torch.cuda.empty_cache()

# Guardar el modelo entrenado
model.save_pretrained("./modelo_urgencia")
tokenizer.save_pretrained("./modelo_urgencia")

# Evaluar el modelo
trainer.evaluate()
