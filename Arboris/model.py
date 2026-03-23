"""
Arboris - Data Preprocessing (Version 2)

Purpose:
- Parse iNat JSON
- Map species labels
- Return dataset
"""

from imports import *
from paths import *

class InatDataset(Dataset):
    def __init__(self):

        with open(TRAIN_JSON, "r") as f:
            data = json.load(f)

        self.images = data["images"]
        self.annotations = data["annotations"]
        self.categories = data["categories"]

        # Map category id → index
        self.cat_map = {cat["id"]: i for i, cat in enumerate(self.categories)}

        # Map image_id → category_id
        self.img_to_cat = {
            ann["image_id"]: ann["category_id"]
            for ann in self.annotations
        }

        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor()
        ])

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img = self.images[idx]

        img_path = os.path.join(TRAIN_DIR, img["file_name"])
        image = Image.open(img_path).convert("RGB")
        image = self.transform(image)

        cat_id = self.img_to_cat.get(img["id"], 0)
        label = self.cat_map.get(cat_id, 0)

        return image, torch.tensor(label)


def get_dataloader():
    dataset = InatDataset()
    return DataLoader(dataset, batch_size=8, shuffle=True)