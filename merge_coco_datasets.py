# =============================================================
# merge_coco_datasets.py
# =============================================================
# Junta múltiplos datasets COCO (já limpos) em um único dataset
# com IDs únicos e categorias padronizadas.
# =============================================================

import json
import os
from pathlib import Path

# Caminhos dos datasets individuais
DATASETS = {
    "pessoas": "data/pessoas",
    "faixas_pedestre": "data/faixas_pedestre",
    "veiculos": "data/veiculos",
    "semaforos": "data/semaforos",
}

# Saída
OUTPUT_DIR = Path("data/merged")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

SPLITS = ["train", "valid", "test"]

# Mapeamento global de categorias
GLOBAL_CATEGORIES = [
    {"id": 1, "name": "pessoa"},
    {"id": 2, "name": "faixa_pedestre"},
    {"id": 3, "name": "veiculo"},
    {"id": 4, "name": "semaforo_verde"},
    {"id": 5, "name": "semaforo_amarelo"},
    {"id": 6, "name": "semaforo_vermelho"},
]

# Mapeamento simples de nome → ID
CAT_NAME_TO_ID = {c["name"]: c["id"] for c in GLOBAL_CATEGORIES}

def merge_split(split):
    merged = {
        "images": [],
        "annotations": [],
        "categories": GLOBAL_CATEGORIES
    }

    next_img_id = 1
    next_ann_id = 1

    for ds_name, ds_path in DATASETS.items():
        json_path = Path(ds_path) / split / "_annotations_clean.coco.json"
        if not json_path.exists():
            print(f"⚠️  JSON não encontrado: {json_path}")
            continue

        with open(json_path, "r", encoding="utf-8") as f:
            coco = json.load(f)

        print(f"🔁 Mesclando {ds_name} ({split}): "
              f"{len(coco['images'])} imgs, {len(coco['annotations'])} anns")

        # Mapeia IDs de imagens antigos → novos
        old_to_new_img = {}
        for img in coco["images"]:
            new_img = img.copy()
            new_img["id"] = next_img_id
            old_to_new_img[img["id"]] = next_img_id
            next_img_id += 1
            merged["images"].append(new_img)

        # Corrige anotações
        for ann in coco["annotations"]:
            new_ann = ann.copy()
            new_ann["id"] = next_ann_id
            new_ann["image_id"] = old_to_new_img.get(ann["image_id"], -1)
            # renumera categoria conforme nome (seguro)
            if "category_id" in ann and "categories" in coco:
                try:
                    cat_name = next(c["name"] for c in coco["categories"] if c["id"] == ann["category_id"])
                    if cat_name in CAT_NAME_TO_ID:
                        new_ann["category_id"] = CAT_NAME_TO_ID[cat_name]
                    else:
                        continue
                except StopIteration:
                    continue
            next_ann_id += 1
            merged["annotations"].append(new_ann)

    # Salva JSON final
    out_path = OUTPUT_DIR / split / "_annotations_merged.coco.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(merged, f, indent=2, ensure_ascii=False)

    print(f"✅ Merge {split} concluído: {len(merged['images'])} imgs, {len(merged['annotations'])} anns")
    return merged


# ===== EXECUÇÃO PRINCIPAL =====
if __name__ == "__main__":
    for split in SPLITS:
        merge_split(split)

    print("\n🎯 Mesclagem concluída com sucesso!")
    print("Arquivos finais em: data/merged/{train, valid, test}/_annotations_merged.coco.json")
