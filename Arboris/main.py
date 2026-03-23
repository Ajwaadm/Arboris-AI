"""
Arboris - Main Entry Point

Purpose:
- Run training or prediction
"""

from paths import create_dirs
from train import train

def main():

    create_dirs()

    print("1. Train")
    choice = input("Enter choice: ")

    if choice == "1":
        train()
    else:
        print("Invalid choice")


if __name__ == "__main__":
    main()