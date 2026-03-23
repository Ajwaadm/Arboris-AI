"""
Arboris - Data Pipeline (Final Version)

Purpose:
- Full taxonomy parsing
- Progressive sampling
"""

from imports import *
from paths import *

class InatDataset(Dataset):
    def __init__(self, fraction=1.0):

        with open(TRAIN_JSON, "r") as f:
            data = json.load(f)

        self.images = data["images"]
        self.annotations = data["annotations"]
        self.categories = data["categories"]

        self.transform = transforms.Compose([
            transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
            transforms.ToTensor()
        ])

        # Build taxonomy maps
        self.tax_maps = {lvl: {} for lvl in TAXONOMY_LEVELS}

        for cat in self.categories:
            for lvl in TAXONOMY_LEVELS:
                if cat[lvl] not in self.tax_maps[lvl]:
                    self.tax_maps[lvl][cat[lvl]] = len(self.tax_maps[lvl])

        self.img_to_cat = {
            ann["image_id"]: ann["category_id"]
            for ann in self.annotations
        }

        self.cat_lookup = {cat["id"]: cat for cat in self.categories}

        # Progressive sampling
        if fraction < 1:
            size = int(len(self.images) * fraction)
            self.images = random.sample(self.images, size)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img = self.images[idx]

        path = os.path.join(TRAIN_DIR, img["file_name"])
        image = Image.open(path).convert("RGB")
        image = self.transform(image)

        cat_id = self.img_to_cat.get(img["id"])
        cat = self.cat_lookup[cat_id]

        labels = [
            self.tax_maps[lvl][cat[lvl]]
            for lvl in TAXONOMY_LEVELS
        ]

        return image, torch.tensor(labels)


def get_dataloader(fraction):
    dataset = InatDataset(fraction)
    return DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True), dataset.tax_maps