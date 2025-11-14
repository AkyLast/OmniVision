import os
import torch
import multiprocessing

def config_gpu():
    config = {
        "cuda_version": torch.version.cuda,
        "cuda_status": torch.cuda.is_available(),
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "Nenhuma"
    }
    return config

def config_workers():
    try:
        import psutil
        physical_cores = psutil.cpu_count(logical=False)
    except ImportError:
        physical_cores = "Não disponível (instale psutil)"

    logical_cores = os.cpu_count()  

    config = {
        "cores_fisics": physical_cores,
        "cores_logics": logical_cores
    }

    return config
