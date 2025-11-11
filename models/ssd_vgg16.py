# =============================================================
# ssd_vgg16.py
# =============================================================
# Este módulo define a função para criação do modelo SSD (Single Shot Detector)
# utilizando a arquitetura base VGG16. O modelo é implementado com PyTorch.
# =============================================================

from torchvision.models import VGG16_Weights
from torchvision.models.detection import ssd300_vgg16

# =============================================================
# Função: create_ssd_model
# =============================================================
# Cria e retorna um modelo SSD300 baseado na arquitetura VGG16.
# O modelo pode ser inicializado com pesos pré-treinados do ImageNet
# para o backbone, enquanto as camadas de detecção são inicializadas do zero.
# =============================================================
def create_ssd_model(num_classes=2):
    """
    Cria um modelo SSD300 com backbone VGG16.

    Parâmetros:
    -----------
    num_classes : int
        Número total de classes a serem detectadas (incluindo o fundo).
        Exemplo: num_classes=2 → [fundo, pessoa]

    Retorna:
    --------
    model : torchvision.models.detection.SSD
        Objeto PyTorch do modelo SSD pronto para treino ou inferência.
    """

    model = ssd300_vgg16(
        weights=None,  # Nenhum peso pré-treinado do COCO é carregado (evita conflito de classes)
        weights_backbone=VGG16_Weights.IMAGENET1K_FEATURES,  # Usa backbone treinado no ImageNet
        num_classes=num_classes  # Define o número de classes do dataset atual
    )

    return model
