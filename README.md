# Image Sampling & Quantization Analysis Tool

This project explores the effect of spatial sampling and intensity quantization on image quality. It calculates MSE, PSNR and SSIM, builds a small dataset across sampling-rate/bit-depth settings, and trains a Linear Regression model to predict SSIM.

## Technologies
Python, OpenCV, NumPy, Pandas, Matplotlib, scikit-image and scikit-learn.

## Run on Windows
1. Install Python 3.10 or newer.
2. Open this folder in VS Code.
3. In the VS Code terminal run:

```powershell
py -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python main.py
```

If PowerShell blocks activation, you can skip activation and run `py -m pip install -r requirements.txt` followed by `py main.py`.

## Output
- `results/`: original/sampled/quantized image previews and the processed image.
- `dataset/image_quality_dataset.csv`: measurements for all sampling-rate and bit-depth combinations.
- Terminal output: MSE, PSNR, SSIM and regression evaluation metrics (R², RMSE and MAE).

## Important notes
- The supplied `images/sample_image.png` is a generated demonstration image so the code runs without extra downloads. It is **not** the original `lena.tiff` image mentioned in the earlier code. If you have permission to use the original image, place it in `images/` and update `IMAGE_PATH` in `main.py`.
- This repository is a reconstruction based on the provided `main.py` and the project report workflow, not the recovered original team repository.
- The generated demo dataset has only 16 configurations. Its regression score should not be presented as proof of the original report's R² > 0.97 result.
