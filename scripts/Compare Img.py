import cv2
import matplotlib.pyplot as plt


def compare_images_positive_diff(image_path1, image_path2, direction="2_minus_1"):
    image1 = cv2.imread(image_path1)
    image2 = cv2.imread(image_path2)

    if not image1 or not image2:
        print("오류: 이미지 경로를 확인하세요.")
        return

    h1, w1 = image1.shape[:2]
    image2 = cv2.resize(image2, (w1, h1))

    image1_rgb = cv2.cvtColor(image1, cv2.COLOR_BGR2RGB)
    image2_rgb = cv2.cvtColor(image2, cv2.COLOR_BGR2RGB)
    if direction == "2_minus_1":
        diff = cv2.subtract(image2_rgb, image1_rgb)
        plot_title = "Positive Difference (Image 2 > Image 1)"
    elif direction == "1_minus_2":
        diff = cv2.subtract(image1_rgb, image2_rgb)
        plot_title = "Positive Difference (Image 1 > Image 2)"
    else:
        print("오류: direction은 '2_minus_1' 또는 '1_minus_2'여야 합니다.")
        return

    r_diff = diff[:, :, 0]
    g_diff = diff[:, :, 1]
    b_diff = diff[:, :, 2]

    total_diff = cv2.cvtColor(diff, cv2.COLOR_RGB2GRAY)
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    fig.suptitle(plot_title, fontsize=16)

    cmap_style = "hot"

    # Red 차이
    im_r = axes[0, 0].imshow(r_diff, cmap=cmap_style, vmin=0, vmax=50)
    axes[0, 0].set_title("Red (R) Channel Difference")
    axes[0, 0].axis("off")
    fig.colorbar(im_r, ax=axes[0, 0], shrink=0.8)

    # Green 차이
    im_g = axes[0, 1].imshow(g_diff, cmap=cmap_style, vmin=0, vmax=50)
    axes[0, 1].set_title("Green (G) Channel Difference")
    axes[0, 1].axis("off")
    fig.colorbar(im_g, ax=axes[0, 1], shrink=0.8)

    # Blue 차이
    im_b = axes[1, 0].imshow(b_diff, cmap=cmap_style, vmin=0, vmax=50)
    axes[1, 0].set_title("Blue (B) Channel Difference")
    axes[1, 0].axis("off")
    fig.colorbar(im_b, ax=axes[1, 0], shrink=0.8)

    # 전체 차이 (R+G+B 합)
    im_total = axes[1, 1].imshow(total_diff, cmap=cmap_style, vmin=0, vmax=50)
    axes[1, 1].set_title("Total (R+G+B) Difference")
    axes[1, 1].axis("off")
    fig.colorbar(im_total, ax=axes[1, 1], shrink=0.8)

    plt.tight_layout(rect=(0, 0.03, 1, 0.95))
    plt.show()


# 비교할 이미지 !!!!!!!
path1 = "GKAN900_meta.png"
path2 = "meerkat.jpg"


print("--- [Image 2 > Image 1] 차이 (추가된 부분) ---")
compare_images_positive_diff(path1, path2, direction="2_minus_1")

print("--- [Image 1 > Image 2] 차이 (제거된 부분) ---")
compare_images_positive_diff(path1, path2, direction="1_minus_2")
