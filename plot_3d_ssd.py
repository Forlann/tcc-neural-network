# =============================================================
# plot_3d_ssd.py
# =============================================================
# Lê o arquivo ssd_3d_metrics.csv e plota gráficos 3D
# de Epochs × Número de Imagens × Métrica (precision / mAP etc.)
# =============================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm

CSV_PATH = "ssd_3d_metrics.csv"

# -------------------------------------------------------------
# Função principal de plotagem 3D
# -------------------------------------------------------------
def plot_3d_metric(csv_file, metric_name="precision"):
    """
    Plota um gráfico 3D com eixos:
    - X: epochs
    - Y: número de imagens
    - Z: métrica escolhida (precision, recall, map50, map50_95)

    metric_name deve ser um dos:
    ["precision", "recall", "map50", "map50_95"]
    """

    df = pd.read_csv(csv_file)

    if metric_name not in ["precision", "recall", "map50", "map50_95"]:
        raise ValueError("Métrica inválida. Escolha: precision, recall, map50, map50_95")

    # pega só as colunas necessárias
    epochs = sorted(df["epoca"].unique())
    imagens = sorted(df["num_imagens"].unique())

    # cria a grade (meshgrid)
    X, Y = np.meshgrid(epochs, imagens)

    # matriz Z (mesma shape da grade)
    Z = np.zeros_like(X, dtype=float)

    # preenchendo Z consultando o CSV
    for i, img_count in enumerate(imagens):
        for j, epc in enumerate(epochs):
            linha = df[(df["num_imagens"] == img_count) & (df["epoca"] == epc)]
            if len(linha) > 0:
                Z[i, j] = linha.iloc[0][metric_name]
            else:
                Z[i, j] = np.nan  # caso falte algum dado

    # ---------------------------------------------------------
    # Plotagem 3D
    # ---------------------------------------------------------
    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection="3d")

    surf = ax.plot_surface(X, Y, Z, cmap=cm.viridis, linewidth=0, antialiased=True)

    ax.set_xlabel("Épocas (Treinamento)", fontsize=12, labelpad=10)
    ax.set_ylabel("Quantidade de Imagens (Dataset)", fontsize=12, labelpad=10)
    ax.set_zlabel(metric_name.upper(), fontsize=12, labelpad=10)

    ax.set_title(f"Evolução da Performance: {metric_name.upper()}\n"
                 f"(Escalabilidade de Dados vs. Tempo de Treino)",
                 fontsize=14)

    fig.colorbar(surf, ax=ax, shrink=0.5, aspect=5)

    ax.view_init(elev=30, azim=225)
    plt.tight_layout()
    plt.show()


# -------------------------------------------------------------
# Se quiser rodar direto
# -------------------------------------------------------------
if __name__ == "__main__":
    # Escolha qual métrica quer plotar:
    # "precision", "recall", "map50", "map50_95"
    plot_3d_metric(CSV_PATH, metric_name="precision")
