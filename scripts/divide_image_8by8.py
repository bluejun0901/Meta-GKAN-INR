import cv2
import numpy as np
import matplotlib.pyplot as plt

BLOCK_SIZE = 64

def load_and_preprocess_image(image_path, block_size=8):
    """1. 이미지를 로드하고 block_sizexblock_size 블록 처리를 위해 패딩을 추가합니다."""
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(f"이미지를 찾을 수 없습니다: {image_path}")

    h, w = img.shape
    # block_sizexblock_size 블록 단위로 나누어지도록 패딩 크기 계산
    h_pad = (block_size - h % block_size) % block_size
    w_pad = (block_size - w % block_size) % block_size
    
    # 부족한 만큼 하단과 우측에 복사 패딩 추가
    img_padded = cv2.copyMakeBorder(img, 0, h_pad, 0, w_pad, cv2.BORDER_REPLICATE)
    return img_padded

def separate_frequencies(img_padded, block_size=8, cutoff=3):
    """2. DCT를 이용해 저주파와 고주파 성분을 분리합니다."""
    h, w = img_padded.shape
    low_freq_img = np.zeros_like(img_padded, dtype=np.float32)
    high_freq_img = np.zeros_like(img_padded, dtype=np.float32)

    # block_sizexblock_size 블록 단위 루프
    for i in range(0, h, block_size):
        for j in range(0, w, block_size):
            block = img_padded[i:i+block_size, j:j+block_size].astype(np.float32)
            
            # DCT 변환
            dct_block = cv2.dct(block)
            
            # 저주파용 마스크 생성 (왼쪽 상단 계수만 추출)
            low_mask = np.zeros((block_size, block_size), dtype=np.float32)
            low_mask[:cutoff, :cutoff] = 1 
            
            # 저주파 복원 (IDCT)
            low_dct = dct_block * low_mask
            low_freq_img[i:i+block_size, j:j+block_size] = cv2.idct(low_dct)
            
            # 고주파 복원 (IDCT) - 원본 DCT 계수에서 저주파를 뺀 나머지
            high_dct = dct_block * (1 - low_mask)
            high_freq_img[i:i+block_size, j:j+block_size] = cv2.idct(high_dct)

    return low_freq_img, high_freq_img

def visualize_results(original, low, high):
    """3. 원본, 저주파, 고주파 이미지를 화면에 출력합니다."""
    images = [original, low, high]
    titles = ["Original (Padded)", "Low Frequency (Smooth)", "High Frequency (Edges)"]
    
    plt.figure(figsize=(18, 6))
    for i in range(3):
        plt.subplot(1, 3, i + 1)
        plt.imshow(images[i], cmap='gray')
        plt.title(titles[i])
        plt.axis('off')
    
    plt.tight_layout()
    plt.savefig("frequency_separation_results.png")

# --- 실행부 ---
if __name__ == "__main__":
    try:
        # 1. 로드
        image_path = 'dataset/STI/Classic/airplane.bmp'  # 실제 파일 경로로 수정해주세요
        padded_img = load_and_preprocess_image(image_path)
        
        # 2. 처리 (cutoff 값 1~BLOCK_SIZE 사이 조절 가능)
        low, high = separate_frequencies(padded_img, block_size=BLOCK_SIZE, cutoff=16)
        
        # 3. 시각화
        visualize_results(padded_img, low, high)
        
    except Exception as e:
        print(f"에러 발생: {e}")