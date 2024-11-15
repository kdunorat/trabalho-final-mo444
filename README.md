# Deepfake Detection Project

This project aims to detect deepfake images using a convolutional neural network (CNN) based on MobileNetV2. The project is implemented in PyTorch and uses a pretrained MobileNetV2 model to extract features, with a custom classifier layer added for deepfake detection.

## Prerequisites

This project requires Python 3.10 and uses Poetry for dependency management. To get started, you will need to have Poetry installed on your system.

## Installation

1. **Clone the Repository**

   ```bash
   git clone <repository-url>
   cd trabalho-final-mo444
   ```

2. **Activate the Poetry Environment**

   Run the following command to enter the Poetry shell:

   ```bash
   poetry shell
   ```

3. **Install Dependencies**

   To install the required dependencies, use the command:

   ```bash
   poetry install
   ```

4. **Set Up the Interpreter**

   Make sure to set up your Python interpreter to use the one provided by Poetry. This is crucial to ensure that all the dependencies are correctly loaded during the development and execution of the project.

## Usage

After installing the dependencies, you can run the training script by executing:

```bash
python main.py
```

Make sure that your dataset is properly organized and the paths are set up in the code before running the training.

## Project Structure

- **main.py**: Entry point for training the model.
- **train_model.py**: Contains the training logic and model evaluation.
- **face_extract.py and train_val_loader/**: Utility functions for data preprocessing, face extraction, etc.
- **README.md**: This document.

## Notes

- This project is intended to be run on a CUDA-compatible GPU to speed up training.
- The model uses MTCNN for face extraction and MobileNetV2 for feature extraction.

## License

This project is licensed under the MIT License.