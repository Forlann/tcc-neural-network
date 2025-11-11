import json
import os

# Root onde você extraiu os 4 zips (cada dataset em sua subpasta)
DATASET_ROOT = "data"  # ajustar se necessário, espera subpastas: data/pessoas, data/veiculos, ...

# listas de nomes de pastas que vamos processar automaticamente
EXPECTED_DATASETS = ["pessoas", "faixas_pedestre", "veiculos", "semaforos"]

# mapeamento final (IDs que você pediu)
GLOBAL_CLASSES = {
    "pessoa": 1,
    "faixa_pedestre": 2,
    "veiculo": 3,
    "semaforo_verde": 4,
    "semaforo_amarelo": 5,
    "semaforo_vermelho": 6,
    "semaforo": 7
}

# Palavras-chave (bem conservadoras)
CLASS_KEYWORDS = {
    "pessoa": ["people", "person", "pessoa", "pedestrian"],
    "faixa_pedestre": ["crosswalk", "cross_walk", "cross-walk", "faixa", "faixa_pedestre"],
    "veiculo": ["car", "truck", "bus", "motorbike", "motorcycle", "vehicle", "van", "auto", "bus_", "truck_"],
    # cores serão mapeadas para semáforo APENAS em datasets de semáforos
    "semaforo_verde": ["green"],
    "semaforo_amarelo": ["yellow", "amber"],
    "semaforo_vermelho": ["red"],
    "semaforo": ["trafficlight", "traffic_light", "trafficlights", "semaforo", "semaforos"]
}

SPLITS = ["train", "valid", "test"]

def find_annotation_json(dataset_folder, split):
    p = os.path.join(dataset_folder, split, "_annotations.coco.json")
    if os.path.exists(p):
        return p
    # alguns Roboflow usam nome com ponto diferente
    p2 = os.path.join(dataset_folder, split, "_annotations.json")
    if os.path.exists(p2):
        return p2
    return None

def map_category_name(orig_name, dataset_key):
    if not isinstance(orig_name, str):
        return None
    name = orig_name.lower().strip()

    # --- exceção específica para dataset de pessoas ---
    if "pesso" in dataset_key or "people" in dataset_key:
        if name in ["0", "people", "person", "pessoa"]:
            return "pessoa"
    # --------------------------------------------------

    # prioridade: exact matches for obvious ones
    for global_name, kws in CLASS_KEYWORDS.items():
        for kw in kws:
            if kw in name:
                # colors -> only map to semaforo colors if dataset is semaforos
                if global_name in ["semaforo_verde","semaforo_amarelo","semaforo_vermelho"]:
                    if "semafor" in dataset_key or "traffic" in dataset_key:
                        return global_name
                    else:
                        continue
                return global_name
    return None

def clean_one_dataset(dataset_key):
    dataset_folder = os.path.join(DATASET_ROOT, dataset_key)
    if not os.path.isdir(dataset_folder):
        print(f"⚠️ Pasta não encontrada: {dataset_folder} — pulando.")
        return

    print(f"\n🔁 Processando dataset: {dataset_key}")

    for split in SPLITS:
        json_path = find_annotation_json(dataset_folder, split)
        if not json_path:
            print(f"  - {split}: sem arquivo JSON encontrado, pulando.")
            continue

        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        orig_categories = [c["name"] for c in data.get("categories", [])]
        print(f"  - {split}: categorias originais ({len(orig_categories)}): {orig_categories[:20]}{'...' if len(orig_categories)>20 else ''}")

        # montar mapeamento old_id -> new_global_id
        id_map = {}
        for cat in data.get("categories", []):
            new_class = map_category_name(cat.get("name",""), dataset_key)
            if new_class:
                id_map[cat["id"]] = GLOBAL_CLASSES[new_class]

        if not id_map:
            print(f"    -> Nenhuma categoria mapeada automaticamente para {dataset_key}/{split}. Verificar manualmente.")
            # salvar cópia com sufixo para análise
            out_path = json_path.replace("_annotations.coco.json", "_annotations_nomapped.coco.json")
            with open(out_path, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=4, ensure_ascii=False)
            print(f"    -> JSON de debug salvo: {out_path}")
            continue

        # filtrar anotações que correspondem a id_map
        new_annotations = []
        used_image_ids = set()
        for ann in data.get("annotations", []):
            old_cat = ann.get("category_id")
            if old_cat in id_map:
                ann["category_id"] = id_map[old_cat]
                new_annotations.append(ann)
                used_image_ids.add(ann["image_id"])

        # filtrar imagens que tem anotações (evitar imagens órfãs)
        new_images = [img for img in data.get("images", []) if img["id"] in used_image_ids]

        # criar categories limpas (somente as usadas)
        used_category_ids = sorted(list(set([ann["category_id"] for ann in new_annotations])))
        new_categories = [{"id": cid, "name": [k for k,v in GLOBAL_CLASSES.items() if v==cid][0], "supercategory":"none"} for cid in used_category_ids]

        print(f"    -> anns originais: {len(data.get('annotations',[]))}, anns mantidas: {len(new_annotations)}")
        print(f"    -> imagens originais: {len(data.get('images',[]))}, imagens mantidas: {len(new_images)}")
        print(f"    -> categorias finais: {[(c['id'],c['name']) for c in new_categories]}")

        # montar novo objeto json
        cleaned = {
            "info": data.get("info", {}),
            "licenses": data.get("licenses", []),
            "images": new_images,
            "annotations": new_annotations,
            "categories": new_categories
        }

        out_json = os.path.join(dataset_folder, split, "_annotations_clean.coco.json")
        with open(out_json, "w", encoding="utf-8") as f:
            json.dump(cleaned, f, indent=4, ensure_ascii=False)
        print(f"    -> _annotations_clean salvo: {out_json}")

def main():
    for ds in EXPECTED_DATASETS:
        clean_one_dataset(ds)
    print("\n✅ Limpeza concluída para todos os datasets (verifique logs acima).")

if __name__ == "__main__":
    main()
