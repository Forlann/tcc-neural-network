import os
import json
import shutil
from pathlib import Path

# Caminhos
DATASETS = {
    "pessoas": "data/pessoas",
    "faixas_pedestre": "data/faixas_pedestre",
    "veiculos": "data/veiculos",
    "semaforos": "data/semaforos",
}
MERGED_DIR = Path("data/merged")
SPLITS = ["train", "valid", "test"]

def find_image(filename):
    """
    Procura a imagem nas pastas dos datasets originais.
    Compatível com arquivos COCO com ou sem 'images/' no caminho.
    """
    for ds_path in DATASETS.values():
        for split in SPLITS:
            # tenta com e sem subpasta 'images'
            img_path_1 = Path(ds_path) / split / "images" / filename
            img_path_2 = Path(ds_path) / split / filename
            if img_path_1.exists():
                return img_path_1
            if img_path_2.exists():
                return img_path_2
    return None


for split in SPLITS:
    merged_json = MERGED_DIR / split / "_annotations_merged.coco.json"
    output_dir = MERGED_DIR / split / "images"
    output_dir.mkdir(parents=True, exist_ok=True)

    with open(merged_json, "r", encoding="utf-8") as f:
        coco = json.load(f)

    missing = 0
    copied = 0

    for img in coco["images"]:
        src = find_image(img["file_name"])
        if src:
            dst = output_dir / src.name
            if not dst.exists():
                shutil.copy(src, dst)
                copied += 1
        else:
            missing += 1

    print(f"✅ {split}: {copied} imagens copiadas, {missing} ausentes.")
