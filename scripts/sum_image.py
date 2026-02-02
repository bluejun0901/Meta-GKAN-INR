import os
import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)) + "/..")
import cv2
import numpy as np
from src.utils import separate_frequencies, pad_image_to_block_size
from skimage.metrics import peak_signal_noise_ratio

def process_image_frequency_sum(folder_path, ref_image_path, block_size=8, cutoff=3, step=50):
    # 1. Reference 이미지 로드 (PSNR 계산용)
    ref_img = cv2.imread(ref_image_path)
    if ref_img is None:
        print(f"Error: Reference 이미지를 찾을 수 없습니다: {ref_image_path}")
        return
    ref_img = pad_image_to_block_size(ref_img, block_size=block_size)

    # 폴더 내 파일 목록 확인 및 인덱스 추출
    file_list = os.listdir(folder_path)

    indices = []
    for f in file_list:
        if f.startswith('low') and f.endswith('.png'):
            # 'low'와 '.png' 사이의 문자열만 추출
            num_part = f.replace('low', '').replace('.png', '')
            
            # 추출된 부분이 숫자인지 확인
            if num_part.isdigit():
                indices.append(int(num_part))

    indices = sorted(indices)

    print(f"{'Index':<10} | {'PSNR (dB)':<15}")
    print("-" * 30)

    for i in indices:
        # i가 step의 배수인 경우만 처리 (0, 50, 100...)
        if i % step != 0:
            continue

        low_path = os.path.join(folder_path, f"low{i}.png")
        high_path = os.path.join(folder_path, f"high{i}.png")

        if not (os.path.exists(low_path) and os.path.exists(high_path)):
            continue

        # 2. 이미지 로드
        img_low_src = cv2.imread(low_path)
        img_high_src = cv2.imread(high_path)

        # 3. 주파수 분리 및 필요한 성분 추출
        # low{i}.png 에서 저주파만 추출
        low_part, _ = separate_frequencies(img_low_src, block_size=block_size, cutoff=cutoff)
        
        # high{i}.png 에서 고주파만 추출
        _, high_part = separate_frequencies(img_high_src, block_size=block_size, cutoff=cutoff)

        # 4. 두 성분 합치기 및 클리핑 (0~255 범위를 벗어날 수 있음)
        sum_img = np.add(low_part, high_part)
        sum_img = np.clip(sum_img, 0, 255).astype(np.uint8)

        # 5. 결과 저장
        save_path = os.path.join(folder_path, f"sum{i}.png")
        cv2.imwrite(save_path, sum_img)

        # 6. PSNR 계산
        # Reference 이미지와 결과 이미지의 크기가 다를 경우를 대비해 resize 처리 (필요시)
        if ref_img.shape != sum_img.shape:
            sum_img_resized = cv2.resize(sum_img, (ref_img.shape[1], ref_img.shape[0]))
        else:
            sum_img_resized = sum_img

        psnr_val = peak_signal_noise_ratio(ref_img, sum_img_resized, data_range=255)
        print(f"{i:<10} | {psnr_val:<15.4f}")

# --- 설정값 ---
target_folder = "runs/2026-02-03_02-16-13/learn/artifacts"      # 이미지가 저장된 폴더
reference_path = "runs/2026-02-03_02-16-13/learn/artifacts/original_padded.png"   # 비교 대상 원본 이미지
BLOCK_SIZE = 64
CUTOFF = 8

# 함수 실행
process_image_frequency_sum(target_folder, reference_path, BLOCK_SIZE, CUTOFF, step=50)