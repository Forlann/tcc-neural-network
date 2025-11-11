# =============================================================
# test_model.py
# =============================================================
# Este script realiza um teste de inicialização do modelo SSD300
# com backbone VGG16, garantindo que ele possa ser instanciado
# e executado corretamente em CPU ou GPU.
# =============================================================

import torch
from models.ssd_vgg16 import create_ssd_model

# =============================================================
# CONFIGURAÇÃO DO DISPOSITIVO
# =============================================================
# Define automaticamente se o modelo usará GPU (CUDA) ou CPU.
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Usando dispositivo:", device)

# =============================================================
# CRIAÇÃO DO MODELO
# =============================================================
# Cria o modelo SSD com 1 classe de interesse + classe de fundo.
model = create_ssd_model(num_classes=2).to(device)
model.eval()  # modo de inferência (desativa gradientes)

# =============================================================
# TESTE DE INFERÊNCIA COM IMAGEM ALEATÓRIA
# =============================================================
# Cria uma imagem de teste aleatória (300x300) apenas para checar a execução.
x = [torch.rand(3, 300, 300).to(device)]

# Desativa cálculo de gradiente para inferência
with torch.no_grad():
    preds = model(x)

# =============================================================
# RESULTADOS DO TESTE
# =============================================================
print("✅ Modelo SSD300_VGG16 carregado e executado com sucesso!")
print("Saídas:", preds[0].keys())
