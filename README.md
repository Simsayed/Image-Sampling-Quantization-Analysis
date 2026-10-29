# Image Sampling & Quantization Analysis Tool

## 1. Project Overview

The **Image Sampling & Quantization Analysis Tool** is a Python-based project that studies how reducing image resolution and pixel intensity levels affects image quality.

The project processes an input image using different sampling rates and quantization bit depths. It compares the processed images with the original image using MSE, PSNR, and SSIM. It also creates a dataset and uses Linear Regression to demonstrate SSIM prediction.

## 2. Objectives

- Understand the effects of image sampling and quantization.
- Reduce image resolution and reconstruct it to its original dimensions.
- Reduce pixel intensity levels using different quantization bit depths.
- Evaluate image quality using MSE, PSNR, and SSIM.
- Generate a dataset for different sampling-rate and bit-depth combinations.
- Demonstrate SSIM prediction using Linear Regression.

## 3. Technologies Used

- **Python:** Main programming language.
- **OpenCV:** Image loading, resizing, and image processing.
- **NumPy:** Numerical calculations and pixel operations.
- **Pandas:** Creating and saving the image-quality dataset.
- **Matplotlib:** Displaying and saving image visualizations.
- **scikit-image:** Calculating the SSIM metric.
- **scikit-learn:** Training and evaluating the Linear Regression model.

## 4. Project Workflow

1. The program searches the `images` folder for a supported image file.
2. It loads the selected image and converts its color format for processing and display.
3. It reduces the image resolution through sampling and resizes it back to the original dimensions.
4. It performs quantization by reducing the number of available pixel intensity levels.
5. It calculates MSE, PSNR, and SSIM to measure image quality.
6. It repeats the processing for four sampling rates and four bit depths.
7. It stores the measurements in a CSV dataset.
8. It trains a Linear Regression model to predict SSIM from sampling rate and bit depth.
9. It displays the regression evaluation metrics and saves image visualizations.

## 5. Image Quality Metrics

### Mean Squared Error (MSE)

MSE measures the average squared difference between the original and processed image pixels. A lower MSE generally indicates less pixel-level error.

### Peak Signal-to-Noise Ratio (PSNR)

PSNR measures image reconstruction quality in decibels (dB). A higher PSNR generally indicates less distortion.

### Structural Similarity Index (SSIM)

SSIM compares the structural similarity between the original and processed images. Values closer to 1 generally indicate greater structural similarity.

## 6. Dataset and Machine Learning

The program tests four sampling rates: 1.0, 0.75, 0.5, and 0.25.

It also tests four bit depths: 8, 6, 4, and 2 bits.

This produces **16 sampling-rate and bit-depth configurations** for the selected input image.

The generated dataset contains these columns:

- `sampling_rate`
- `bit_depth`
- `mse`
- `psnr`
- `ssim`

Linear Regression uses `sampling_rate` and `bit_depth` as input features to predict SSIM. The data is split into training and testing sets, and the model is evaluated using R², RMSE, and MAE.

The dataset is small, so the regression results are intended as a demonstration rather than proof of general predictive performance.

## 7. Project Structure

```text
Image-Sampling-Quantization-GitHub/
├── images/                  # Input images
├── dataset/
│   └── image_quality_dataset.csv
├── results/
│   ├── original_image.png
│   ├── sampled_image.png
│   ├── quantized_image.png
│   ├── comparison.png
│   └── processed.png
├── main.py                  # Main Python program
├── requirements.txt         # Required Python libraries
├── .gitignore
└── README.md
```

The `dataset` and `results` files are generated or updated when the program runs.

## 8. Installation and Execution on Windows

### Requirements

- Python 3.10 or newer
- Visual Studio Code
- Python extension for VS Code (recommended)

### Step 1: Open the project

Open the project folder in Visual Studio Code and open its terminal.

### Step 2: Create a virtual environment

```powershell
py -m venv .venv
```

### Step 3: Install the required libraries

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

### Step 4: Add an input image

Place at least one supported image inside the `images` folder. Supported extensions are `.tiff`, `.tif`, `.png`, `.jpg`, `.jpeg`, and `.bmp`.

The program automatically selects an available image. If `sample_image.png` exists, it is preferred; otherwise, the program selects the first supported filename in alphabetical order.

### Step 5: Run the program

```powershell
.\.venv\Scripts\python.exe main.py
```

## 9. Output

After successful execution, the program:

- Displays and saves the original, sampled, quantized, and comparison images.
- Saves the processed image in the `results` folder.
- Generates `dataset/image_quality_dataset.csv`.
- Prints MSE, PSNR, and SSIM values.
- Prints the Linear Regression evaluation metrics: R², RMSE, and MAE.

## 10. Limitations

- The dataset contains only 16 configurations for the selected image.
- Regression performance can vary depending on the input image and the train-test split.
- The Linear Regression result should not be treated as proof of performance on unseen images.
- The results depend on the selected input image and processing settings.

## 11. Conclusion

This project demonstrates how image sampling and quantization affect image quality. It combines image processing, quality measurement, dataset generation, visualization, and a basic machine-learning experiment in one Python application.
