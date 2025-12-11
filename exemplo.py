import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import cm
import numpy as np

def plotar_grafico_3d_csv():
    # =============================================================================
    # 1. CARREGAR E PREPARAR OS DADOS
    # =============================================================================
    
    nome_arquivo = 'resultados_3d_yolo_parcial.csv' # Certifique-se que o nome está correto
    
    try:
        df = pd.read_csv(nome_arquivo)
        # Remove espaços em branco dos nomes das colunas (caso existam)
        df.columns = [c.strip() for c in df.columns]
        print("Colunas encontradas:", df.columns)
    except FileNotFoundError:
        print(f"ERRO: O arquivo '{nome_arquivo}' não foi encontrado na pasta.")
        return

    # --- CONFIGURAÇÃO IMPORTANTE ---
    # Qual coluna usar para o Eixo Z? (Olhando seu print: 'precision', 'mAP50' ou 'mAP50-95')
    COLUNA_METRICA = 'precision' 
    
    # Qual o tamanho total do seu dataset (100%)? 
    # O gráfico vai usar isso para converter a "porcentagem" em "número de imagens"
    TOTAL_IMAGENS_REAL = 17302 

    # =============================================================================
    # 2. TRANSFORMAR LISTA EM MATRIZ (PIVOT)
    # =============================================================================
    
    # Cria uma tabela dinâmica (Pivot Table)
    # Linhas (Index) = dataset_porcentagem
    # Colunas = epoca
    # Valores = A métrica escolhida (precision)
    pivot_df = df.pivot(index='dataset_porcentagem', columns='epoca', values=COLUNA_METRICA)
    
    # Preenche valores vazios com 0 (caso algum treino tenha falhado) para não quebrar o gráfico
    pivot_df = pivot_df.fillna(0)

    # Ordena os dados para garantir que o gráfico não fique "riscado"
    pivot_df = pivot_df.sort_index(ascending=True) # Ordena Y (10%, 20%...)
    pivot_df = pivot_df.sort_index(axis=1, ascending=True) # Ordena X (Epoca 10, 20...)

    # =============================================================================
    # 3. CRIAR OS EIXOS PARA O MATPLOTLIB
    # =============================================================================

    # Eixo X: As épocas (são os cabeçalhos das colunas da tabela pivot)
    epochs = pivot_df.columns.values
    
    # Eixo Y: As porcentagens (são o índice da tabela pivot)
    # Vamos converter a porcentagem (ex: 10) para numero de imagens (ex: 1600)
    porcentagens = pivot_df.index.values
    imagens = (porcentagens / 100) * TOTAL_IMAGENS_REAL

    # Cria a malha (Meshgrid) necessária para superfícies 3D
    X, Y = np.meshgrid(epochs, imagens)
    
    # Eixo Z: Os valores da métrica (a matriz de valores)
    Z = pivot_df.values

    # =============================================================================
    # 4. PLOTAGEM
    # =============================================================================
    
    fig = plt.figure(figsize=(14, 9))
    ax = fig.add_subplot(111, projection='3d')

    # Cria a superfície
    # cmap='viridis' (verde/azul/amarelo) ou 'plasma' (roxo/laranja) ou 'coolwarm'
    surf = ax.plot_surface(X, Y, Z, cmap=cm.viridis, linewidth=0.1, edgecolor='k', alpha=0.9, antialiased=True)

    # --- Estética e Etiquetas ---
    ax.set_xlabel('Épocas (Treinamento)', fontsize=11, labelpad=10, fontweight='bold')
    ax.set_ylabel('Qtd. Imagens (Dataset)', fontsize=11, labelpad=10, fontweight='bold')
    
    nome_metrica_formatado = COLUNA_METRICA.replace('metrics/', '').upper()
    ax.set_zlabel(f'Métrica: {nome_metrica_formatado}', fontsize=11, labelpad=10, fontweight='bold')
    
    ax.set_title(f'Evolução da Performance: {nome_metrica_formatado}\n(Escalabilidade de Dados vs. Tempo de Treino)', fontsize=14)

    # Adiciona barra de cores
    cbar = fig.colorbar(surf, ax=ax, shrink=0.5, aspect=10, pad=0.1)
    cbar.set_label(nome_metrica_formatado, rotation=270, labelpad=15)

    # Ajusta limites do Z para ficar bonito (entre 0 e 1 se for precisão)
    ax.set_zlim(0, 1.0)

    # Ajusta ângulo de visão inicial
    ax.view_init(elev=30, azim=230)

    print("Gerando gráfico...")
    plt.show()

if __name__ == "__main__":
    plotar_grafico_3d_csv()