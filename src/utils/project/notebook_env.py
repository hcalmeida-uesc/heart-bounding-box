"""Notebook bootstrap helpers for exploratory notebooks."""

from __future__ import annotations

import os
import sys
from pathlib import Path


def configure_notebook_environment(chdir_to_src: bool = True) -> Path:
    """Add src to sys.path and optionally switch cwd to the src directory.

    Returns:
        Path to the repository root.
    """
    repo_root = Path(__file__).resolve().parents[3]
    src_dir = repo_root / "src"

    src_dir_str = str(src_dir)
    if src_dir_str not in sys.path:
        sys.path.insert(0, src_dir_str)

    if chdir_to_src:
        os.chdir(src_dir)
    return repo_root


def resolve_existing_path(
    env_var: str,
    candidates: list[Path],
    description: str,
) -> Path:
    """Resolve a dataset/output path from an env var or known local candidates."""
    env_value = os.environ.get(env_var)
    if env_value:
        path = Path(env_value).expanduser()
        if path.exists():
            return path
        raise FileNotFoundError(f"{description} definido em {env_var} não existe: {path}")

    resolved = next((path for path in candidates if path.exists()), None)
    if resolved is not None:
        return resolved

    candidate_text = "\n".join(f"- {path}" for path in candidates)
    raise FileNotFoundError(
        f"Nenhum caminho encontrado para {description}. "
        f"Exporte {env_var} ou ajuste os candidatos:\n{candidate_text}"
    )


def resolve_imagecas_base_path() -> Path:
    """Resolve o diretório ImageCAS com arquivos ``*.img.nii.gz``."""
    return resolve_existing_path(
        "IMAGECAS_BASE_PATH",
        [
            Path("/media/matheus/HD/DatasetsCCTA/ImageCAS/1-1000"),
            Path("/data04/home/mpmaia/ImageCAS/database/1-1000"),
            Path("/home/matheus/DatasetsCCTA/ImageCAS/1-1000")
        ],
        "ImageCAS",
    )


def resolve_processed_imagecas_path() -> Path:
    """Resolve o diretório de caches/resultados processados do ImageCAS."""
    return resolve_existing_path(
        "IMAGECAS_PROCESSED_PATH",
        [
            Path("/media/matheus/HD/DatasetsCCTA/Processed_ImageCAS"),
            Path("/data04/home/mpmaia/ImageCAS/Processed_ImageCAS"),
        ],
        "Processed ImageCAS",
    )


