import os

def remap_classes_especificas(diretorio_labels, mapa_de_classes):
    """
    Percorre um diretório de labels YOLO (.txt) e substitui os IDs
    de classe com base em um dicionário de mapeamento.

    Args:
        diretorio_labels (str): O caminho para a pasta com os .txt
        mapa_de_classes (dict): Dicionário no formato {'ID_antigo': 'ID_novo'}
    """
    print(f"--- Iniciando remapeamento em: {diretorio_labels} ---")
    
    # Verifica se o diretório existe
    if not os.path.isdir(diretorio_labels):
        print(f"ERRO: O diretório não existe. Pulando: {diretorio_labels}")
        print("---------------------------------------------------\n")
        return

    arquivos_modificados = 0
    total_detecções_remapeadas = 0

    # IDs que esperamos encontrar (como strings)
    classes_esperadas = set(mapa_de_classes.keys())

    for nome_arquivo in os.listdir(diretorio_labels):
        if nome_arquivo.endswith(".txt"):
            caminho_arquivo = os.path.join(diretorio_labels, nome_arquivo)
            novas_linhas = []
            arquivo_modificado_nesta_execucao = False
            
            try:
                with open(caminho_arquivo, 'r') as f:
                    linhas = f.readlines()
                
                if not linhas:
                    continue  # Pula arquivos vazios

                for linha in linhas:
                    partes = linha.strip().split()
                    
                    if len(partes) >= 5:
                        id_antigo_str = partes[0]
                        
                        # A MÁGICA: Verifica se o ID antigo está no nosso mapa
                        if id_antigo_str in mapa_de_classes:
                            # Se sim, substitui pelo novo ID
                            id_novo_str = mapa_de_classes[id_antigo_str]
                            partes[0] = id_novo_str
                            
                            novas_linhas.append(" ".join(partes) + "\n")
                            total_detecções_remapeadas += 1
                            arquivo_modificado_nesta_execucao = True
                        else:
                            # Se o ID não está no mapa (ex: 3, 4, 5...)
                            # mantém a linha original e avisa
                            print(f"AVISO: ID de classe '{id_antigo_str}' não esperado "
                                  f"encontrado em {nome_arquivo}. Linha mantida.")
                            novas_linhas.append(linha)
                    else:
                        novas_linhas.append(linha) # Mantém linhas vazias/inválidas
                
                # Reescreve o arquivo com as classes remapeadas
                if arquivo_modificado_nesta_execucao:
                    with open(caminho_arquivo, 'w') as f:
                        f.writelines(novas_linhas)
                    arquivos_modificados += 1

            except Exception as e:
                print(f"Erro ao processar o arquivo {caminho_arquivo}: {e}")

    print(f"Remapeamento concluído.")
    print(f"Total de arquivos .txt modificados: {arquivos_modificados}")
    print(f"Total de detecções remapeadas: {total_detecções_remapeadas}")
    print("---------------------------------------------------\n")

# =============================================================================
# 1. CONFIGURE SEU MAPEAMENTO E CAMINHOS AQUI
# =============================================================================

# ⚠️ FAÇA BACKUP ANTES DE EXECUTAR

# Mapeamento desejado:
# 'green'  (0) -> 2
# 'red'    (1) -> 3
# 'yellow' (2) -> 4
MAPEAMENTO_SEMAFOROS = {
    '0': '2',
    '1': '3',
    '2': '4'
}

# String de placeholder para checagem
PLACEHOLDER = "COLOQUE_O_CAMINHO_PARA"

# Coloque os caminhos para o SEU dataset de semáforos
# Exemplo: "D:/Datasets/semaforos.yolov11/train/labels"
PATH_TREINO_LABELS = "C:/Users/leonardo/Desktop/tcc/semaforos.yolov11/train/labels"
PATH_VALID_LABELS = "C:/Users/leonardo/Desktop/tcc/semaforos.yolov11/valid/labels"
PATH_TEST_LABELS = "C:/Users/leonardo/Desktop/tcc/semaforos.yolov11/test/labels"


# =============================================================================
# 2. EXECUÇÃO DO SCRIPT
# =============================================================================

print("=== INICIANDO REMAPEAMENTO DOS SEMÁFOROS ===")

# Processa a pasta de treino
if PLACEHOLDER not in PATH_TREINO_LABELS:
    remap_classes_especificas(PATH_TREINO_LABELS, MAPEAMENTO_SEMAFOROS)
else:
    print("AVISO: Caminho de TREINO não configurado. Pulando...")

# Processa a pasta de validação
if PLACEHOLDER not in PATH_VALID_LABELS:
    remap_classes_especificas(PATH_VALID_LABELS, MAPEAMENTO_SEMAFOROS)
else:
    print("AVISO: Caminho de VALIDAÇÃO não configurado. Pulando...")

# Processa a pasta de teste
if PLACEHOLDER not in PATH_TEST_LABELS:
    remap_classes_especificas(PATH_TEST_LABELS, MAPEAMENTO_SEMAFOROS)
else:
    print("AVISO: Caminho de TESTE não configurado. Pulando...")