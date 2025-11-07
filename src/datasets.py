import os
import shutil
from roboflow import Roboflow

# API keys e identificadores
api_key = os.getenv("ROBOFLOW_API_KEY")
workspace_key = os.getenv("ROBOFLOW_WORKSPACE_KEY")
project_key = os.getenv("ROBOFLOW_DATASET_KEY")

# Pasta de destino
output_dir = "data/roboflow"
os.makedirs(output_dir, exist_ok=True)

# Inicializa o Roboflow e baixa o dataset
rf = Roboflow(api_key=api_key)
dataset = (
    rf
    .workspace(workspace_key)
    .project(project_key)
    .version(5)  # altere conforme a versão desejada
    .download("yolov8")
)

# Caminho do dataset baixado
download_path = dataset.location

# Caminho final desejado
final_path = os.path.join(output_dir, os.path.basename(download_path))

# Se já existir, remove a antiga antes de mover
if os.path.exists(final_path):
    shutil.rmtree(final_path)

# Move a pasta baixada para o destino
shutil.move(download_path, final_path)

print(f"✅ Dataset movido com sucesso para: {final_path}")
