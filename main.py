# main.py
import tkinter as tk
from gui import App
from model import Model
import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

def main():
    app = App()
    app.mainloop()

if __name__ == "__main__":
    main()