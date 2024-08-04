# This program is a demo of how to use our model

import tkinter as tk
from tkinter import filedialog
from PIL import Image, ImageTk
import cv2
import numpy as np
import torch

# set the global variables
WINDOW_SIZE = "844x532"
IMAGE_SIZE = (512, 512)
PAD = 7


def on_label_click(event):
    # TODO: 当label被点击时，显示原来的图骗你
    pass


def on_label_release(event):
    # TODO: 当label被释放时，显示处理后的图像
    pass


def pad_resize_image(image: np.ndarray):
    # pad the image to make it square and resize it to the default size
    h, w = image.shape
    if h > w:
        pad = (h - w) // 2
        new_image = np.ones((h, h), dtype=np.uint8) * 255
        new_image[:, pad:pad + w] = image
    else:
        pad = (w - h) // 2
        new_image = np.ones((w, w), dtype=np.uint8) * 255
        new_image[pad:pad + h, :] = image
    new_image = cv2.resize(new_image, IMAGE_SIZE)
    return new_image


def open_file():
    file_path = filedialog.askopenfilename(
        title='Open a file',
        filetypes=[('Image files', '*.png *.jpg *.jpeg'), ]
    )
    if file_path:
        image = cv2.imread(file_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        image = pad_resize_image(image)
        image = Image.fromarray(image)
        photo = ImageTk.PhotoImage(image)

        label.config(image=photo)
        label.image = photo


def set_default_image():
    image = cv2.imread('loading.jpg')
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    image = pad_resize_image(image)
    image = Image.fromarray(image)
    photo = ImageTk.PhotoImage(image)

    label.config(image=photo)
    label.image = photo


def on_scale_iter_change(value):
    # TODO: 当迭代次数改变时，更新对原图进行卷积并更新图像
    pass


def on_scale_factor_x_change(value):
    # TODO: 当X方向缩放比例改变时，更新卷积核并更新图像
    pass


def on_scale_factor_y_change(value):
    # TODO: 当Y方向缩放比例改变时，更新卷积核并更新图像
    pass


def add_pixel():
    # TODO: 绑定：当鼠标左键按下时，添加像素
    pass


def remove_pixel():
    # TODO: 绑定：当鼠标右键按下时，删除像素
    pass


def predict():
    # TODO: 调用模型进行预测
    pass


# create the main window
root = tk.Tk()
root.title('GUI Example')
# root.geometry(WINDOW_SIZE)
root.resizable(False, False)
root.option_add('*font', ('Microsoft YaHei UI', 10))

# create a label to display the image
label = tk.Label(root, bg="#CCCCCC", padx=PAD, pady=PAD)
set_default_image()
label.pack(side=tk.LEFT, fill=tk.BOTH, expand=False)
label.bind("<Button-1>", on_label_click)
label.bind("<ButtonRelease-1>", on_label_release)

# create a control panel
control_frame = tk.Frame(root, padx=PAD, pady=PAD, bg="#CCCCCC")
control_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=False)

# slogan
slogan = tk.Label(control_frame, text='RECOGNIZE THE TEXT!', font=('Microsoft YaHei UI', 14), bg="#CCCCCC")
slogan.grid(row=0, column=0, padx=PAD, pady=PAD, columnspan=2)

# create a button for opening a file
open_btn = tk.Button(control_frame, text='Open File', command=open_file)
open_btn.grid(row=1, column=0, padx=PAD, pady=PAD, sticky='ew')
clear_btn = tk.Button(control_frame, text='Clear', command=set_default_image)
clear_btn.grid(row=1, column=1, padx=PAD, pady=PAD, sticky='ew')

# create a set of buttons and scales for setting parameters
scale_iter = tk.Scale(control_frame, from_=1, to=20, orient='horizontal', label='Number of Iterations',
                      length=220, resolution=1)
scale_iter.grid(row=2, column=0, padx=PAD, pady=PAD, sticky='ew', columnspan=2)
scale_iter.set(10)

scale_factor_x = tk.Scale(control_frame, from_=0, to=1, orient='horizontal', label='X Factor',
                          length=100, resolution=0.1)
scale_factor_x.grid(row=3, column=0, padx=PAD, pady=PAD, sticky='ew')
scale_factor_x.set(0.5)
scale_factor_y = tk.Scale(control_frame, from_=0, to=1, orient='horizontal', label='Y Factor',
                          length=100, resolution=0.1)
scale_factor_y.grid(row=3, column=1, padx=PAD, pady=PAD, sticky='ew')
scale_factor_y.set(0.5)

add_pixel_btn = tk.Button(control_frame, text='Add Pixel', command=add_pixel)
add_pixel_btn.grid(row=4, column=0, padx=PAD, pady=PAD, sticky='ew')
remove_pixel_btn = tk.Button(control_frame, text='Remove Pixel', command=remove_pixel)
remove_pixel_btn.grid(row=4, column=1, padx=PAD, pady=PAD, sticky='ew')

# create a button for predicting
predict_btn = tk.Button(control_frame, text='EXTRACT WORDS!', command=predict)
predict_btn.grid(row=5, column=0, padx=PAD, pady=PAD, columnspan=2, sticky='ew')

root.mainloop()
