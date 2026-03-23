"""
Arboris - Plotting (Final Version)

Purpose:
- Plot loss over time
"""

from imports import *
from paths import *

def plot():

    if not os.path.exists(LOG_FILE):
        print("No logs found")
        return

    df = pd.read_csv(LOG_FILE)

    plt.plot(df["loss"])
    plt.title("Training Loss")
    plt.savefig(OUTPUT_DIR / "loss.png")

    print("Plot saved")

if __name__ == "__main__":
    plot()