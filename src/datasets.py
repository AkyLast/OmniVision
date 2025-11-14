import os
import json
import shutil
from roboflow import Roboflow

api_key = os.getenv("ROBOFLOW_API_KEY")
workspace_key = os.getenv("ROBOFLOW_WORKSPACE_KEY")
project_key = os.getenv("ROBOFLOW_DATASET_KEY")

MODO_CLEAN = False

with open("config.json", "r", encoding="utf-8") as file:
    CONFIG = json.load(file)

DATASET_CONFIG = CONFIG["datasets_config"]

names = ["ROBOFLOW_API_KEY", "ROBOFLOW_WORKSPACE_KEY", "ROBOFLOW_DATASET_KEY"]
values = [api_key, workspace_key, project_key]
status = True

for i, key in enumerate(values):
    if key == '':
        print("Erro! chave não encontrada:", names[i])
        status = False

if status:
    try:

        output_dir = "data/roboflow" if not MODO_CLEAN else "data/raw"
        os.makedirs(output_dir, exist_ok=True)

        rf = Roboflow(api_key=api_key)
        dataset = (
            rf
            .workspace(DATASET_CONFIG["roboflow_workspace"])
            .project(DATASET_CONFIG["roboflow_dataset_name"])
            .version(DATASET_CONFIG["dataset_version"])  
            .download(DATASET_CONFIG["download_format_model"])
        )

        download_path = dataset.location

        final_path = os.path.join(output_dir, os.path.basename(download_path))

        if os.path.exists(final_path):
            shutil.rmtree(final_path)

        shutil.move(download_path, final_path)

        print(f"✅ Dataset movido com sucesso para: {final_path}")

    except Exception as e:
        print("Erro ao baixar o dataset:", e)
