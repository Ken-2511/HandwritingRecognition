# Handwriting Recognition System

This repository implements a comprehensive **Handwriting Recognition System** that allows users to recognize handwritten text from images using both traditional machine learning methods (SVM) and deep learning models (CRNN, Faster R-CNN).

## Features:
- **Support Vector Machine (SVM)** for character-based handwriting recognition.
- **Convolutional Recurrent Neural Network (CRNN)** for word-level handwriting recognition.
- **Faster R-CNN** for detecting text regions in images.
- **Graphic User Interface (GUI)** to easily interact with the models and perform handwriting recognition.

### Directory Structure
- **Baseline Model**: Contains SVM-based handwriting recognition models and feature extraction methods.
- **Train**: Files related to training deep learning models (CRNN, Faster R-CNN).
- **Graphic User Interface**: Contains the code for the GUI where users can load images and recognize handwriting.
  
---

## Getting Started

### Prerequisites

To run the project, you'll need the following:

- **Python 3.x**
- Install required Python packages using `pip`:

    ```bash
    pip install -r requirements.txt
    ```

### Running the GUI

To use the handwriting recognition system through a graphical interface, follow these steps:

1. **Launch the GUI**:
    Run the `main.py` script from the `Graphic User Interface` folder:

    ```bash
    python Graphic\ User\ Interface/main.py
    ```

2. **Using the GUI**:
    - Once the GUI is launched, you'll be able to upload an image of handwritten text.
    - The system will process the image, segment the text, and run the CRNN model to output the recognized text.
    - The output will be displayed within the GUI window.

### Model Training

If you'd like to train the models yourself:

1. **SVM**:
    - Navigate to the `Baseline Model` directory and run the respective scripts (`SVM.py`) for training the SVM model.

2. **CRNN**:
    - Use the `CRNN_test.ipynb` notebook under the `Train/others` folder to train the CRNN model for word recognition.

3. **Faster R-CNN**:
    - Train the Faster R-CNN model by running the `Faster_R_CNN.ipynb` notebook in the `Train/others` folder.

---

### Example Usage

1. **Upload Handwritten Image**: Load a scanned image of handwritten text using the GUI.
2. **Preprocessing**: The system will preprocess the image (binarization, feature extraction, segmentation).
3. **Recognition**: The model (SVM or CRNN) will process the image and output the recognized text.
4. **View Results**: The recognized text will be displayed on the screen.

---

### Contributions

Feel free to contribute to this project by submitting issues or pull requests!

---

### License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
