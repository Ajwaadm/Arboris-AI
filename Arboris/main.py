"""
Arboris - Main (Final Version)

Purpose:
- CLI entry
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
        print("Invalid")

if __name__ == "__main__":
    main()