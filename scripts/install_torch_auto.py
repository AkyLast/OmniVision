import os
import subprocess
import platform
import shutil

def run(cmd):
    """Executa comando shell e retorna stdout (ou None em erro)."""
    try:
        return subprocess.check_output(cmd, shell=True, stderr=subprocess.DEVNULL).decode().strip()
    except subprocess.CalledProcessError:
        return None

def install(cmd):
    print(f"\nInstalando: {cmd}\n")
    os.system(cmd)

def has_nvidia_gpu():
    """Detecta se há GPU NVIDIA disponível."""
    return shutil.which("nvidia-smi") is not None

def main():
    system = platform.system().lower()
    arch = platform.machine().lower()

    print(f"Sistema detectado: {system} ({arch})")

    if not has_nvidia_gpu():
        print("Nenhuma GPU NVIDIA detectada — instalando PyTorch CPU.")
        install("uv pip install torch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1")
        return

    driver_version = run("nvidia-smi --query-gpu=driver_version --format=csv,noheader")
    cuda_tag = "cu121"  

    if driver_version:
        print(f"Driver NVIDIA detectado: {driver_version}")
        if "12.4" in driver_version:
            cuda_tag = "cu124"
        elif "12.1" in driver_version or "12" in driver_version:
            cuda_tag = "cu121"
        else:
            print(f"Driver {driver_version} não reconhecido — usando CUDA 12.1 por padrão.")
    else:
        print("GPU detectada, mas não foi possível ler o driver. Usando CUDA 12.1.")

    print(f"Instalando PyTorch com suporte a {cuda_tag.upper()}...")

    index_url = f"https://download.pytorch.org/whl/{cuda_tag}"
    cmd = (
        f"uv pip install "
        f"torch==2.5.1+{cuda_tag} "
        f"torchvision==0.20.1+{cuda_tag} "
        f"torchaudio==2.5.1+{cuda_tag} "
        f"--extra-index-url {index_url}"
    )
    install(cmd)

    print("\nTestando instalação...")
    os.system('uv run python -c "import torch; print(torch.__version__, torch.version.cuda, torch.cuda.is_available())"')

if __name__ == "__main__":
    main()
