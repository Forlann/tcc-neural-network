import torch

def detection_collate_fn(batch):
    """
    Collate function para modelos de detecção do torchvision.
    
    'batch' é uma lista de tuplas: [(img1, target1), (img2, target2), ...]
    
    Esta função deve retornar:
    1. Um tensor de imagens empilhadas (ex: [batch_size, 3, 416, 416])
    2. UMA LISTA de dicionários 'target' (ex: [target1, target2, ...])
    """
    
    # A função zip(*) "descompacta" a lista de tuplas em duas tuplas:
    # (img1, img2, ...), (target1, target2, ...)
    images, targets = zip(*batch)
    
    # O 'images' já vem como Tensor por causa do 'transform.ToTensor()'
    # 'torch.stack' junta a tupla de tensores em um único tensor
    images = torch.stack(images, 0)
    
    # 'targets' deve ser retornado como uma LISTA de dicionários.
    # O modelo sabe lidar com isso.
    return images, list(targets)