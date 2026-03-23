"""
Arboris - Training (Version 2)

Purpose:
- Train model
- Save checkpoints
- Log metrics
"""

from imports import *
from model import CNNModel
from preprocess import get_dataloader
from paths import *

def train():

    loader = get_dataloader()
    num_classes = 1000  # simplified

    model = CNNModel(num_classes).to(DEVICE)

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.CrossEntropyLoss()

    best_loss = float("inf")

    logs = []

    for epoch in range(2):

        model.train()
        total_loss = 0

        for images, labels in loader:

            images = images.to(DEVICE)
            labels = labels.to(DEVICE)

            optimizer.zero_grad()

            outputs = model(images)
            loss = criterion(outputs, labels)

            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        avg_loss = total_loss / len(loader)
        print(f"Epoch {epoch+1} Loss: {avg_loss:.4f}")

        logs.append({
            "epoch": epoch + 1,
            "train_loss": avg_loss
        })

        # Save checkpoint
        torch.save(
            model.state_dict(),
            CHECKPOINT_DIR / "latest.pt"
        )

        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save(
                model.state_dict(),
                CHECKPOINT_DIR / "best.pt"
            )

    # Save logs
    df = pd.DataFrame(logs)
    df.to_csv(LOG_FILE, index=False)