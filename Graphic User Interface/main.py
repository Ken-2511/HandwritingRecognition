# This program is a demo of how to use our model

import tkinter as tk
from tkinter import ttk
from tkinter import messagebox
from tkinter import filedialog
from PIL import Image, ImageTk

def open_file():
    file_path = filedialog.askopenfilename(
        title='Open a file',
        filetypes=[('Image files', '*.png *.jpg *.jpeg'),]
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

# Create the main window
root = tk.Tk()
root.title('GUI Example')
root.geometry('600x400')

# create a label to display the image
label = ttk.Label(root)
label.grid(row=0, column=0, rowspan=4)

# Create a button
button = ttk.Button(root, text='Open File', command=open_file)
button.grid(row=0, column=1)

root.mainloop()