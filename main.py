
"""Image Sampling and Quantization Analysis Tool.

Automatically selects an available image from the images folder.
Calculates MSE, PSNR and SSIM, creates a dataset, and evaluates
a Linear Regression model for SSIM prediction.
"""

from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from skimage.metrics import structural_similarity as ssim
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error


ROOT = Path(__file__).resolve().parent
IMAGES_DIR = ROOT / "images"
RESULTS_DIR = ROOT / "results"
DATASET_DIR = ROOT / "dataset"

RESULTS_DIR.mkdir(exist_ok=True)
DATASET_DIR.mkdir(exist_ok=True)

SUPPORTED_EXTENSIONS = {".tiff", ".tif", ".png", ".jpg", ".jpeg", ".bmp"}


def find_image():
    """Find an image automatically inside the images folder."""
    if not IMAGES_DIR.exists():
        raise FileNotFoundError(
            f"Images folder not found: {IMAGES_DIR}"
        )

    # Prefer the original demo filename if it still exists.
    preferred = IMAGES_DIR / "sample_image.png"
    if preferred.is_file():
        return preferred

    # Otherwise select the first supported image by filename.
    image_files = sorted(
        (
            path for path in IMAGES_DIR.iterdir()
            if path.is_file() and path.suffix.lower() in SUPPORTED_EXTENSIONS
        ),
        key=lambda path: path.name.lower()
    )

    if not image_files:
        raise FileNotFoundError(
            f"No supported images found in {IMAGES_DIR}. "
            "Add a .tiff, .tif, .png, .jpg, .jpeg or .bmp image."
        )

    return image_files[0]


def load_image(path):
    """Load an image using OpenCV and convert BGR to RGB."""
    image = cv2.imread(str(path), cv2.IMREAD_COLOR)

    if image is None:
        raise ValueError(
            f"OpenCV could not read the image: {path}. "
            "Check whether the file is valid."
        )

    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


def sample_image(image, scale):
    """Downsample an image and reconstruct it to its original dimensions."""
    if not 0 < scale <= 1:
        raise ValueError("Scale must be greater than 0 and at most 1.")

    height, width = image.shape[:2]
    new_height = max(1, int(height * scale))
    new_width = max(1, int(width * scale))

    sampled = cv2.resize(
        image, (new_width, new_height),
        interpolation=cv2.INTER_AREA
    )

    return cv2.resize(
        sampled, (width, height),
        interpolation=cv2.INTER_LINEAR
    )


def quantize_image(image, bits):
    """Reduce pixel intensity levels according to the selected bit depth."""
    if not 1 <= bits <= 8:
        raise ValueError("Bit depth must be between 1 and 8.")

    levels = 2 ** bits
    factor = 256 / levels

    quantized = np.floor(
        image.astype(np.float32) / factor
    ) * factor

    return np.clip(quantized, 0, 255).astype(np.uint8)


def mse(original, processed):
    """Calculate Mean Squared Error."""
    difference = (
        original.astype(np.float64)
        - processed.astype(np.float64)
    )
    return float(np.mean(difference ** 2))


def psnr(original, processed):
    """Calculate Peak Signal-to-Noise Ratio in decibels."""
    mse_value = mse(original, processed)

    if mse_value == 0:
        return float("inf")

    return float(20 * np.log10(255.0 / np.sqrt(mse_value)))


def ssim_score(original, processed):
    """Calculate Structural Similarity Index."""
    return float(
        ssim(original, processed, channel_axis=2, data_range=255)
    )


def save_plot(image, title, path):
    """Display an image and save it to the results folder."""
    plt.figure(figsize=(7, 5))
    plt.imshow(image)
    plt.title(title)
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.show()
    plt.close()





def main():
    # Automatically locate an existing image.
    image_path = find_image()
    print("Using image:", image_path.name)

    image = load_image(image_path)
    print("Original image shape:", image.shape)

    save_plot(
        image, "Original Image",
        RESULTS_DIR / "original_image.png"
    )

    sampled = sample_image(image, 0.5)
    save_plot(
        sampled, "Sampled Image (50% resolution)",
        RESULTS_DIR / "sampled_image.png"
    )

    quantized = quantize_image(image, 4)
    save_plot(
        quantized, "Quantized Image (4 bits)",
        RESULTS_DIR / "quantized_image.png"
    )

    processed = quantize_image(sampled, 4)
    comparison = np.concatenate([image, processed], axis=1)

    save_plot(
        comparison, "Original vs Sampled and Quantized",
        RESULTS_DIR / "comparison.png"
    )

    # Calculate image-quality metrics.
    mse_value = mse(image, processed)
    psnr_value = psnr(image, processed)
    ssim_value = ssim_score(image, processed)

    print("\nImage Quality Metrics")
    print("---------------------")
    print(f"MSE : {mse_value:.4f}")
    print(f"PSNR: {psnr_value:.4f} dB")
    print(f"SSIM: {ssim_value:.6f}")

    cv2.imwrite(
        str(RESULTS_DIR / "processed.png"),
        cv2.cvtColor(processed, cv2.COLOR_RGB2BGR)
    )

    # Generate a dataset for different sampling rates and bit depths.
    sampling_rates = [1.0, 0.75, 0.5, 0.25]
    bit_depths = [8, 6, 4, 2]
    rows = []

    for rate in sampling_rates:
        sampled_image = sample_image(image, rate)

        for bits in bit_depths:
            result = quantize_image(sampled_image, bits)

            rows.append({
                "sampling_rate": rate,
                "bit_depth": bits,
                "mse": mse(image, result),
                "psnr": psnr(image, result),
                "ssim": ssim_score(image, result)
            })

    df = pd.DataFrame(rows)

    print("\nGenerated Image-Quality Dataset")
    print("--------------------------------")
    print(df.round(4).to_string(index=False))

    csv_path = DATASET_DIR / "image_quality_dataset.csv"
    df.to_csv(csv_path, index=False)
    print("\nDataset saved to:", csv_path)

    # Demonstration of SSIM prediction using Linear Regression.
    X = df[["sampling_rate", "bit_depth"]]
    y = df["ssim"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42
    )

    model = LinearRegression()
    model.fit(X_train, y_train)

    predictions = model.predict(X_test)

    print("\nLinear Regression Evaluation")
    print("----------------------------")
    print(f"R²  : {r2_score(y_test, predictions):.4f}")
    print(
        f"RMSE: {np.sqrt(mean_squared_error(y_test, predictions)):.4f}"
    )
    print(f"MAE : {mean_absolute_error(y_test, predictions):.4f}")
    print("Predicted SSIM:", np.round(predictions, 4))

    print("\nAll processing completed successfully.")
    print("Check the results folder for generated images.")
    print("Check the dataset folder for the CSV file.")
    print(
        "Note: Results depend on the selected input image and "
        "do not necessarily match the original team's reported results."
    )


if __name__ == "__main__":
    main()
