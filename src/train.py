import os
import json
import torch
from ultralytics import YOLO
from utils import config_gpu, config_workers

with open("config.json", "r", encoding="utf-8") as file:
    CONFIG = json.load(file)

MODEL_CONFIG = CONFIG["model_config"]
PROJECT_DIR = "experiments"
EXPERIMENT_NAME = "structral_v0"

gpu = config_gpu()

device = "cuda" if gpu["cuda_status"] else "cpu"
print(f"Usando dispositivo: {device}")

# --- 2. Caminhos dos dados ---
data_yaml = f"{MODEL_CONFIG["dataset_path"]}/data.yaml"

# --- 3. Carregar modelo base ---
model = YOLO(f"models/pretrained/{MODEL_CONFIG["model_base"]}") 

# --- 4. Treinar ---
results = model.train(
    data=data_yaml,     
    epochs=10,          
    imgsz=640,         
    batch=8,            
    device=device,      
    project=PROJECT_DIR,  
    name=EXPERIMENT_NAME,  
)

# --- 5. Validar ---
#metrics = model.val()  # avalia automaticamente o modelo final
#print("Métricas de validação:", metrics)

# --- 6. Inferência de teste ---
