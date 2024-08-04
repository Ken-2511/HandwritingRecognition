# This program is a demo of how to use our model

import tkinter as tk
from tkinter import messagebox
from tkinter import filedialog
from PIL import Image, ImageTk


def open_file():
    file_path = filedialog.askopenfilename(
        title='Open a file',
        filetypes=[('Image files', '*.png *.jpg *.jpeg'), ]
    )
    if file_path:
        image = Image.open(file_path)
        image.thumbnail((400, 400))
        photo = ImageTk.PhotoImage(image)

        label.config(image=photo)
        label.image = photo


def set_parameters():
    pass


def predict():
    pass


# create the main window
root = tk.Tk()
root.title('GUI Example')
root.geometry('600x400')

# create a label to display the image
label = tk.Label(root, width=100, height=100)
label.grid(row=0, column=0, rowspan=4)

# create a control panel
control_frame = tk.Frame(root)
control_frame.grid(row=0, column=1, sticky='nw')

# create a button
button = tk.Button(control_frame, text='Open File', command=open_file)
button.grid(row=0, column=0)

root.mainloop()
