env_vars = {
    "ROBOFLOW_API_KEY": "",
    "ROBOFLOW_WORKSPACE_KEY": "",
    "ROBOFLOW_DATASET_KEY": ""
}

with open(".env", "w") as f:
    for key, value in env_vars.items():
        f.write(f"{key}={value}\n")
