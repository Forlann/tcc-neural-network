from dataset import COCODataset, get_transform

dataset = COCODataset(
    root="data/merged/train/images",
    annotation_file="data/merged/train/_annotations_merged.coco.json",
    transforms=get_transform(train=True)
)

print(f"Total de imagens: {len(dataset)}")

img, target = dataset[0]
print("Shape da imagem:", img.shape)
print("Caixas:", target["boxes"].shape)
print("Labels:", target["labels"].unique())
