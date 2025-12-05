import os
import json
import torch
from ultralytics import YOLO
from utils import config_gpu, config_workers

with open("config.json", "r", encoding="utf-8") as file:
    CONFIG = json.load(file)

MODEL_CONFIG = CONFIG["model_config"]
PROJECT_DIR = MODEL_CONFIG["PROJECT_DIR"]
count = len(os.listdir(PROJECT_DIR))
num = f"0{count}" if count < 10 else count
EXPERIMENT_NAME = f'{MODEL_CONFIG["model_train"]["experimentations"]}{num}'

gpu = config_gpu()
workers = config_workers()
device = "cuda" if gpu["cuda_status"] else "cpu"

data_yaml = f'{MODEL_CONFIG["dataset_path"]}/data.yaml'
model_usage = f'models/pretrained/{MODEL_CONFIG["model_base"]}'

mensagem = f"""
    -=- Resumo da Configuração -=-
-------------------------------------
Caminho do data.yaml:           {data_yaml.split('/')[-2]}
Modelo base a ser utilizado:    {model_usage.split('/')[-1]}
Projeto:                        {PROJECT_DIR}
Total de itens na pasta:        {num}
Nome da experimentação:         {EXPERIMENT_NAME}

--- GPU ---
CUDA disponível:                {gpu["cuda_status"]}
Versão CUDA:                    {gpu["cuda_version"]}
GPU detectada:                  {gpu["gpu"]}

--- CPU / Workers ---
Cores físicos:                  {workers["cores_fisics"]}
Cores lógicos:                  {workers["cores_logics"]}

--- Dispositivo ---
Dispositivo em uso:             {device}
-------------------------------------
"""

print(mensagem)


model = YOLO(model_usage) 
BATCH = MODEL_CONFIG["model_train"]["batch"]
IMGSZ = MODEL_CONFIG["model_train"]["imgsz"]

if __name__ == "__main__":
    results = model.train(
        data=data_yaml,     
        epochs=1,          
        imgsz=IMGSZ,         
        batch=BATCH,            
        device=device,      
        project=PROJECT_DIR,  
        name=EXPERIMENT_NAME,  
    )