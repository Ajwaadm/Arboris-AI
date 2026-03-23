"""
Arboris - Plotting (Version 2)

Purpose:
- Plot training loss
"""

from imports import *
from paths import *

def plot_loss():

    if not os.path.exists(LOG_FILE):
        print("No log file found")
        return

    df = pd.read_csv(LOG_FILE)

    plt.plot(df["epoch"], df["train_loss"])
    plt.title("Training Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")

    save_path = OUTPUT_DIR / "loss.png"
    plt.savefig(save_path)

    print(f"Saved: {save_path}")


if __name__ == "__main__":
    plot_loss()