import tkinter as tk
from src.GUI.gui import ControlVisualisation


if __name__ == '__main__':
    root_app = tk.Tk()
    app = ControlVisualisation(root_app)
    root_app.mainloop()
