import importlib.metadata
import importlib.util
import os
import pathlib
import platform
import subprocess
import sys

import folder_paths
import torch
from huggingface_hub import snapshot_download


NATTEN_WHEEL_INDEX = "https://whl.natten.org"
NATTEN_BUILDS = {
    ("2.5", "12.1"): "0.17.5+torch250cu121",
    ("2.7", "12.6"): "0.21.0+torch270cu126",
}


def ensure_download_directory(path):
    try:
        os.makedirs(path, exist_ok=True)
    except PermissionError as error:
        raise PermissionError(
            f"PMRF cannot create its model directory: {path}. "
            "Make the ComfyUI runtime user the owner of this directory before startup."
        ) from error


def ensure_models():
    pmrf_path = os.path.join(folder_paths.models_dir, "pmrf")
    pmrf_model_path = os.path.join(pmrf_path, "model.safetensors")
    pmrf_config_path = os.path.join(pmrf_path, "config.json")

    if not (os.path.isfile(pmrf_model_path) and os.path.isfile(pmrf_config_path)):
        print("[PMRF] Downloading restoration model...")
        ensure_download_directory(pmrf_path)
        snapshot_download(
            repo_id="ohayonguy/PMRF_blind_face_image_restoration",
            allow_patterns=["model.safetensors", "config.json"],
            local_dir=pmrf_path,
        )

    upscale_models_path = os.path.join(folder_paths.models_dir, "upscale_models")
    ensure_download_directory(upscale_models_path)
    for model_name in ("RealESRGAN_x2plus.pth", "RealESRGAN_x4plus.pth"):
        model_path = os.path.join(upscale_models_path, model_name)
        if not os.path.isfile(model_path):
            print(f"[PMRF] Downloading {model_name}...")
            snapshot_download(
                repo_id="2kpr/Real-ESRGAN",
                allow_patterns=model_name,
                local_dir=upscale_models_path,
            )


def patch_basicsr():
    basicsr_spec = importlib.util.find_spec("basicsr")
    if basicsr_spec is None or basicsr_spec.origin is None:
        return

    path = pathlib.Path(basicsr_spec.origin).parent / "data" / "degradations.py"
    if not path.exists():
        return

    old_import = "from torchvision.transforms.functional_tensor import rgb_to_grayscale"
    new_import = "from torchvision.transforms.functional import rgb_to_grayscale"
    content = path.read_text(encoding="utf-8")
    if old_import in content:
        print("[PMRF] Patching BasicSR for modern torchvision...")
        path.write_text(content.replace(old_import, new_import), encoding="utf-8")


def ensure_natten():
    torch_version = torch.__version__.split("+", 1)[0]
    torch_series = ".".join(torch_version.split(".")[:2])
    cuda_version = torch.version.cuda

    if platform.system() != "Linux" or platform.machine() not in {"x86_64", "AMD64"}:
        raise RuntimeError(
            "This PMRF fork currently provides automatic NATTEN installation only "
            "for Linux x86_64."
        )
    if sys.version_info[:2] not in {(3, 9), (3, 10), (3, 11), (3, 12)}:
        raise RuntimeError(
            "The supported NATTEN wheels require Python 3.9-3.12; found "
            f"{sys.version_info.major}.{sys.version_info.minor}."
        )

    natten_version = NATTEN_BUILDS.get((torch_series, cuda_version))
    if natten_version is None:
        supported = ", ".join(
            f"Torch {torch_key} / CUDA {cuda_key}"
            for torch_key, cuda_key in NATTEN_BUILDS
        )
        raise RuntimeError(
            f"No tested NATTEN build is configured for torch=={torch.__version__} "
            f"with CUDA {cuda_version}. Supported environments: {supported}."
        )

    try:
        installed_version = importlib.metadata.version("natten")
    except importlib.metadata.PackageNotFoundError:
        installed_version = None

    if installed_version == natten_version:
        print(f"[PMRF] NATTEN {installed_version} is already installed.")
        return

    print(f"[PMRF] Installing NATTEN {natten_version}...")
    subprocess.check_call(
        [
            sys.executable,
            "-m",
            "pip",
            "install",
            "--no-deps",
            "--force-reinstall",
            f"natten=={natten_version}",
            "--find-links",
            NATTEN_WHEEL_INDEX,
        ]
    )


ensure_models()
patch_basicsr()
ensure_natten()
