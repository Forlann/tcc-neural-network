# =============================================================
# visualize_predictions.py
# =============================================================
# Este script realiza a inferência (teste) de imagens usando o
# modelo SSD treinado e exibe visualmente as detecções de objetos.
# =============================================================

import torch
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image
import os

from models.ssd_vgg16 import create_ssd_model
from dataset import COCODataset, get_transform  # usa o dataset que converte labels YOLO

# =============================================================
# CONFIGURAÇÕES
# =============================================================
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODEL_PATH = "ssd_vgg16_merged.pth"  # caminho do modelo treinado
DATASET_PATH = "data/merged/test/images"  # caminho das imagens de teste
THRESHOLD = 0.5  # confiança mínima para exibir uma detecção
CLASS_NAMES = [
    "background",
    "pessoa",
    "faixa_pedestre",
    "veiculo",
    "semaforo_verde",
    "semaforo_amarelo",
    "semaforo_vermelho",
]


# =============================================================
# CARREGAR O MODELO
# =============================================================
model = create_ssd_model(num_classes=7)  # 6 classes + background
model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
model.to(DEVICE)
model.eval()

print("✅ Modelo SSD carregado e pronto para inferência.")

# =============================================================
# FUNÇÃO PARA VISUALIZAR DETECÇÕES
# =============================================================
def visualize_prediction(image_path):
    # Abre imagem
    img = Image.open(image_path).convert("RGB")

    # Transforma em tensor
    transform = get_transform(train=False)
    img_tensor = transform(img).unsqueeze(0).to(DEVICE)

    # Realiza a inferência
    with torch.no_grad():
        preds = model(img_tensor)[0]

    # Extrai predições
    boxes = preds["boxes"].cpu().numpy()
    scores = preds["scores"].cpu().numpy()

    # Cria figura
    fig, ax = plt.subplots(1, figsize=(10, 10))
    ax.imshow(img)

    # Desenha cada caixa detectada acima do threshold
    for box, score, label in zip(boxes, scores, preds["labels"].cpu().numpy()):
        if score > THRESHOLD:
            x1, y1, x2, y2 = box
            rect = patches.Rectangle((x1, y1), x2 - x1, y2 - y1,
                                 linewidth=2, edgecolor="lime", facecolor="none")
            ax.add_patch(rect)
            class_name = CLASS_NAMES[label] if label < len(CLASS_NAMES) else f"id_{label}"
            ax.text(
                x1, y1 - 5, f"{class_name} {score:.2f}",
                color="yellow", fontsize=10, backgroundcolor="black"
        )


    plt.title(f"Detecções SSD (score > {THRESHOLD})")
    plt.axis("off")
    plt.show()

# =============================================================
# EXECUTAR TESTE EM UMA IMAGEM
# =============================================================
# Escolhe uma imagem do dataset
test_images = [f for f in os.listdir(DATASET_PATH) if f.endswith(".jpg")]
if not test_images:
    print("⚠️ Nenhuma imagem encontrada na pasta de teste.")
else:
    image_path = os.path.join(DATASET_PATH, test_images[350])  # pega a primeira imagem
    print(f"Exibindo predições para: {image_path}")
    visualize_prediction(image_path)
