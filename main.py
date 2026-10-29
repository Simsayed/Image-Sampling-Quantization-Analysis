"""Image Sampling and Quantization Analysis Tool.

Reconstructed from the user's supplied main.py. The included sample image is a
synthetic demo image, not the original Lena/USC-SIPI image used in the original
project. Replace IMAGE_PATH with your original dataset image if you have it.
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
IMAGE_PATH = ROOT / "images" / "sample_image.png"
RESULTS_DIR = ROOT / "results"
DATASET_DIR = ROOT / "dataset"
RESULTS_DIR.mkdir(exist_ok=True)
DATASET_DIR.mkdir(exist_ok=True)


def load_image(path):
    """Read image using OpenCV and convert BGR to RGB."""
    image = cv2.imread(str(path))
    if image is None:
        raise FileNotFoundError(
            f"Could not load image: {path}. Put an image in the images folder and update IMAGE_PATH."
        )
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


def sample_image(image, scale):
    """Downsample by scale, then reconstruct to the original size."""
    if not 0 < scale <= 1:
        raise ValueError("scale must be greater than 0 and at most 1")
    height, width = image.shape[:2]
    new_height = max(1, int(height * scale))
    new_width = max(1, int(width * scale))
    sampled = cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_AREA)
    return cv2.resize(sampled, (width, height), interpolation=cv2.INTER_LINEAR)


def quantize_image(image, bits):
    """Reduce each channel to 2**bits discrete intensity levels."""
    if not 1 <= bits <= 8:
        raise ValueError("bits must be between 1 and 8")
    levels = 2 ** bits
    factor = 256 / levels
    quantized = np.floor(image.astype(np.float32) / factor) * factor
    return np.clip(quantized, 0, 255).astype(np.uint8)


def mse(original, compressed):
    """Mean squared error; cast to float to avoid uint8 subtraction wraparound."""
    difference = original.astype(np.float64) - compressed.astype(np.float64)
    return float(np.mean(difference ** 2))


def psnr(original, compressed):
    mse_value = mse(original, compressed)
    if mse_value == 0:
        return float("inf")
    return float(20 * np.log10(255.0 / np.sqrt(mse_value)))


def ssim_score(original, compressed):
    return float(ssim(original, compressed, channel_axis=2, data_range=255))


def save_plot(image, title, path):
    plt.figure(figsize=(6, 5))
    plt.imshow(image)
    plt.title(title)
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.show()
    plt.close()


def main():
    image = load_image(IMAGE_PATH)
    print("Original image shape:", image.shape)
    save_plot(image, "Original Image", RESULTS_DIR / "original_image.png")

    sampled_image = sample_image(image, 0.5)
    save_plot(sampled_image, "Sampled Image (50% resolution)", RESULTS_DIR / "sampled_image.png")

    quantized_image = quantize_image(image, 4)
    save_plot(quantized_image, "Quantized Image (4 bits)", RESULTS_DIR / "quantized_image.png")

    processed_image = quantize_image(sample_image(image, 0.5), 4)
    comparison = np.concatenate([image, processed_image], axis=1)
    save_plot(comparison, "Original (left) vs Sampled + Quantized (right)", RESULTS_DIR / "comparison.png")

    mse_val = mse(image, processed_image)
    psnr_val = psnr(image, processed_image)
    ssim_val = ssim_score(image, processed_image)
    print("\nImage Quality Metrics")
    print("---------------------")
    print(f"MSE : {mse_val:.4f}")
    print(f"PSNR: {psnr_val:.4f} dB")
    print(f"SSIM: {ssim_val:.6f}")
    cv2.imwrite(str(RESULTS_DIR / "processed.png"), cv2.cvtColor(processed_image, cv2.COLOR_RGB2BGR))

    sampling_rates = [1.0, 0.75, 0.5, 0.25]
    bit_depths = [8, 6, 4, 2]
    data = []
    for rate in sampling_rates:
        for bits in bit_depths:
            processed = quantize_image(sample_image(image, rate), bits)
            data.append([
                rate, bits, mse(image, processed), psnr(image, processed), ssim_score(image, processed)
            ])
    df = pd.DataFrame(data, columns=["sampling_rate", "bit_depth", "mse", "psnr", "ssim"])
    print("\nGenerated image-quality dataset")
    print(df.round(4).to_string(index=False))
    csv_path = DATASET_DIR / "image_quality_dataset.csv"
    df.to_csv(csv_path, index=False)
    print("\nDataset saved to:", csv_path)

    # The 16-row dataset is small, so the score is a demonstration, not proof of generalization.
    X = df[["sampling_rate", "bit_depth"]]
    y = df["ssim"]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42
    )
    model = LinearRegression()
    model.fit(X_train, y_train)
    prediction = model.predict(X_test)
    print("\nLinear Regression evaluation")
    print("-----------------------------")
    print("R²  :", round(r2_score(y_test, prediction), 4))
    print("RMSE:", round(mean_squared_error(y_test, prediction) ** 0.5, 4))
    print("MAE :", round(mean_absolute_error(y_test, prediction), 4))
    print("Predicted SSIM values:", np.round(prediction, 4))
    print("\nNote: These results are from the included demo image. They are not the original team's reported results.")

if __name__ == "__main__":
    main()
