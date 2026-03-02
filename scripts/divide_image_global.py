import cv2
import matplotlib.pyplot as plt
import numpy as np


def load_global_image(image_path):
    """이미지를 로드하고 DCT를 위해 크기를 짝수로 맞춥니다."""
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError("이미지를 로드할 수 없습니다.")

    # cv2.dct는 가로, 세로 크기가 모두 짝수여야 합니다.
    h, w = img.shape
    new_h = h if h % 2 == 0 else h + 1
    new_w = w if w % 2 == 0 else w + 1
    img_resized = cv2.resize(img, (new_w, new_h))

    return img_resized.astype(np.float32)


def separate_global_frequencies(img_float, low_ratio=0.1):
    """이미지 전체에 DCT를 적용하여 저주파와 고주파를 분리합니다."""
    # 1. 전역 DCT 수행
    dct_coeffs = cv2.dct(img_float)

    h, w = dct_coeffs.shape
    low_mask = np.zeros((h, w), dtype=np.float32)

    # 2. 저주파 영역 설정 (왼쪽 상단 일부)
    # 이미지 크기에 비례하여 cutoff 지점을 결정합니다.
    cutoff_h = int(h * low_ratio)
    cutoff_w = int(w * low_ratio)
    low_mask[:cutoff_h, :cutoff_w] = 1

    # 3. 역변환 (IDCT)
    # 저주파만 남기기
    low_freq_dct = dct_coeffs * low_mask
    low_freq_img = cv2.idct(low_freq_dct)

    # 고주파만 남기기 (전체 - 저주파)
    high_freq_dct = dct_coeffs * (1 - low_mask)
    high_freq_img = cv2.idct(high_freq_dct)

    return low_freq_img, high_freq_img


def visualize_global_results(original, low, high):
    plt.figure(figsize=(15, 5))
    titles = ["Original (Global)", "Low Frequency (Global)", "High Frequency (Global)"]
    imgs = [original, low, high]

    for i in range(3):
        plt.subplot(1, 3, i + 1)
        plt.imshow(imgs[i], cmap="gray")
        plt.title(titles[i])
        plt.axis("off")
    plt.savefig("global_frequency_separation.png")


# 실행
img_global = load_global_image("dataset/STI/Classic/airplane.bmp")
low_global, high_global = separate_global_frequencies(img_global, low_ratio=0.1)
visualize_global_results(img_global, low_global, high_global)
