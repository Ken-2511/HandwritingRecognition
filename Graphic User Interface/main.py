# This program is a demo of how to use our model

import tkinter as tk
from tkinter import filedialog
from PIL import Image, ImageTk
import cv2
import numpy as np
import seg_utils
import torch
import CRNN

# set the global variables
WINDOW_SIZE = "844x532"
IMAGE_SIZE = (512, 512)
PAD = 7


class TextImage:
    # this class is used to store the images and their concomitant information
    def __init__(self, image: np.ndarray):
        self.original_image = image
        self.processed_image = image.copy()
        self.lower_bound = None
        self.upper_bound = None
        self.calculate_lower_upper_bounds()
        self.is_processed = False

    def calculate_lower_upper_bounds(self):
        sorted_img = np.sort(self.original_image.reshape(-1))
        self.lower_bound = sorted_img[int(0.01 * len(sorted_img))].item()
        self.upper_bound = sorted_img[int(0.1 * len(sorted_img))].item()

    def set_original_image(self, image):
        raise ValueError('The original image cannot be changed. If you want to change the image, '
                         'please create a new TextImage object.')

    def set_processed_image(self, image):
        if image.shape != self.original_image.shape:
            raise ValueError('The shape of the image is not the same as the original image: '
                             f'{image.shape} vs {self.original_image.shape}')
        self.processed_image = image
        self.is_processed = True

    def get_original_image_tk(self):
        image = Image.fromarray(self.original_image)
        image = ImageTk.PhotoImage(image)
        return image

    def get_processed_image_tk(self):
        image = Image.fromarray(self.processed_image)
        image = ImageTk.PhotoImage(image)
        return image

    def get_processed_image_tensor(self):
        image = self.processed_image.copy()
        image = np.array(image > 127, dtype=np.uint8)
        return torch.from_numpy(image).unsqueeze(0)


# set a global variable to store the image
_image = None


def on_label_click(event):
    global _image
    if _image is None:
        return
    assert isinstance(_image, TextImage)
    photo = _image.get_original_image_tk()
    label.config(image=photo)
    label.image = photo


def on_label_release(event):
    global _image
    if _image is None:
        return
    assert isinstance(_image, TextImage)
    photo = _image.get_processed_image_tk()
    label.config(image=photo)
    label.image = photo


def pad_resize_image(image: np.ndarray):
    # pad the image to make it square and resize it to the default size
    h, w = image.shape
    # the padding color is the mean of the image (excluding the 15% lowest)
    pixels = np.sort(image.reshape(-1))
    pad_value = np.mean(image.reshape(-1)[pixels[int(0.15 * len(pixels)):]])
    if h > w:
        pad = (h - w) // 2
        new_image = np.ones((h, h), dtype=np.uint8) * pad_value
        new_image[:, pad:pad + w] = image
    else:
        pad = (w - h) // 2
        new_image = np.ones((w, w), dtype=np.uint8) * pad_value
        new_image[pad:pad + h, :] = image
    new_image = cv2.resize(new_image, IMAGE_SIZE).astype(np.uint8)
    return new_image


def open_file():
    file_path = filedialog.askopenfilename(
        title='Open a file',
        filetypes=[('Image files', '*.png *.jpg *.jpeg'), ]
    )
    if file_path:
        image = cv2.imread(file_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        image = pad_resize_image(image).astype(np.uint8)
        global _image
        _image = TextImage(image)
        photo = _image.get_original_image_tk()
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

    global _image
    _image = None


# set the global variable of the conv kernel
_conv_iteration = 10
_conv_factor_y = 0.5
_conv_factor_x = 0.5
_conv = torch.nn.Conv2d(1, 1, 3, padding=1, bias=False)


def update_conv_img():
    global _image, _conv, _conv_iteration
    if _image is None:
        return
    assert isinstance(_image, TextImage)
    image = _image.original_image.copy()
    image = torch.from_numpy(image).float().unsqueeze(0)
    # new_img = seg_utils.non_maximum_suppression(image, _image.lower_bound, _image.upper_bound,
                                                # _conv_iteration, _conv)
    new_img = seg_utils.non_maximum_suppression(image, _image.lower_bound*1.5, _image.upper_bound,
                                                _conv_iteration)
    new_img = new_img.squeeze(0).numpy() * 255
    new_img = new_img.astype(np.uint8)
    _image.set_processed_image(new_img)
    photo = _image.get_processed_image_tk()
    label.config(image=photo)
    label.image = photo


def update_conv_kernel():
    global _conv, _conv_factor_y, _conv_factor_x
    default_value = 2
    weight = np.ones((3, 3), dtype=np.float32) * default_value
    weight[0, 0] = default_value * _conv_factor_y * _conv_factor_x
    weight[0, 2] = default_value * _conv_factor_y * _conv_factor_x
    weight[2, 0] = default_value * _conv_factor_y * _conv_factor_x
    weight[2, 2] = default_value * _conv_factor_y * _conv_factor_x
    weight[0, 1] = default_value * _conv_factor_y
    weight[2, 1] = default_value * _conv_factor_y
    weight[1, 0] = default_value * _conv_factor_x
    weight[1, 2] = default_value * _conv_factor_x
    _conv.weight.data = torch.tensor([[weight.tolist()]], dtype=torch.float32)


def on_scale_iter_change(value):
    global _conv_iteration
    _conv_iteration = int(value)
    update_conv_img()


def on_scale_factor_x_change(value):
    global _conv_factor_y
    _conv_factor_y = float(value)
    update_conv_kernel()
    update_conv_img()


def on_scale_factor_y_change(value):
    global _conv_factor_x
    _conv_factor_x = float(value)
    update_conv_kernel()
    update_conv_img()


def add_pixel():
    # TODO: 绑定：当鼠标左键按下时，添加像素
    pass


def remove_pixel():
    # TODO: 绑定：当鼠标右键按下时，删除像素
    pass


def predict():
    # TODO: 调用模型进行预测
    if _image is None:
        return
    assert isinstance(_image, TextImage)
    if not _image.is_processed:
        update_conv_img()

    # initialize the model
    model = CRNN.CRNN()
    model_name = CRNN.get_model_name(33, "Aug")
    model_path = "../Machine_Learning_Output/CRNN/"
    model.load_state_dict(torch.load(model_path + model_name, weights_only=True))
    model.to("cuda:0")
    model.eval()

    ans = dict()
    original_img = torch.from_numpy(_image.original_image).float().unsqueeze(0)
    ans["original"] = original_img
    processed_img = _image.get_processed_image_tensor()
    # processed_img = processed_img[:, ::2, ::2]  # make the image smaller
    all_same_group = seg_utils.find_all_connected_pixels(processed_img)
    img = original_img.clone()
    bias = _conv_iteration / 3
    rectangles = seg_utils.find_min_rectangle(all_same_group, bias)
    # rectangles = [[y1 * 2, x1 * 2, y2 * 2, x2 * 2] for x1, y1, x2, y2 in rectangles]
    rectangles = [[y1, x1, y2, x2] for x1, y1, x2, y2 in rectangles]
    rectangles = [[round(x) for x in rect] for rect in rectangles]
    rectangles, rows = seg_utils.sort_rectangles(rectangles)
    ans["rectangles"] = rectangles
    ans["rows"] = rows

    # 接下来是识别单词
    words = []
    for row in rows:
        batch = []
        for rect in row.rectangles:
            word_img = seg_utils.get_word_image(img, rect)
            batch.append(word_img)
        batch = torch.stack(batch).to("cuda:0")
        with torch.no_grad():
            output, probabilities = model.forward_beam(batch)
        output = model.beam_output_to_words(output)
        output = [x[0] for x in output]
        row.add_words(output)
        words.extend(output)
        words.append('\n')
    ans["word_list"] = words

    # 整理成一个字符串
    paragraph = ""
    for word in words:
        paragraph += word + " "
    ans["paragraph"] = paragraph

    # show the processed image
    image = _image.original_image.copy()
    for row in rows:
        cv2.line(image, (0, int(row.y_mean)), (image.shape[1], int(row.y_mean)), (0, 0, 0), 1)
        for word, rect in zip(row.words, row.rectangles):
            y1, x1, y2, x2 = rect
            cv2.rectangle(image, (x1, y1), (x2, y2), (0, 0, 0), 1)
            cv2.putText(image, word, (x1, y1-5), cv2.FONT_HERSHEY_SIMPLEX,
                        0.5, (0, 0, 0), 1)
    _image.set_processed_image(image)
    photo = _image.get_processed_image_tk()
    label.config(image=photo)
    label.image = photo

    print(ans["paragraph"])


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
scale_iter = tk.Scale(control_frame, from_=0, to=20, orient='horizontal', label='Number of Iterations',
                      length=220, resolution=1, command=on_scale_iter_change)
scale_iter.grid(row=2, column=0, padx=PAD, pady=PAD, sticky='ew', columnspan=2)
scale_iter.set(10)

scale_factor_x = tk.Scale(control_frame, from_=0, to=1, orient='horizontal', label='X Factor',
                          length=100, resolution=0.01, command=on_scale_factor_x_change)
scale_factor_x.grid(row=3, column=0, padx=PAD, pady=PAD, sticky='ew')
scale_factor_x.set(0.5)
scale_factor_y = tk.Scale(control_frame, from_=0, to=1, orient='horizontal', label='Y Factor',
                          length=100, resolution=0.01, command=on_scale_factor_y_change)
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
