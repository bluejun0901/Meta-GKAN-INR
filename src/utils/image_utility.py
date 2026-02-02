import numpy as np
import cv2

def pad_image_to_block_size(img, block_size=8):
    # 채널 유무에 관계없이 높이와 너비 추출
    h, w = img.shape[:2]

    h_pad = (block_size - h % block_size) % block_size
    w_pad = (block_size - w % block_size) % block_size
    
    # cv2.copyMakeBorder는 다채널 이미지도 자동으로 처리합니다.
    img_padded = cv2.copyMakeBorder(img, 0, h_pad, 0, w_pad, cv2.BORDER_REPLICATE)
    return img_padded

def separate_frequencies(img_padded, block_size=8, cutoff=3) -> tuple[np.ndarray, np.ndarray]:
    # 이미지의 형태 확인 (H, W) 또는 (H, W, C)
    if img_padded.ndim == 3:
        h, w, c = img_padded.shape
        # img_padded = cv2.cvtColor(img_padded, cv2.COLOR_BGR2YCrCb)
    else:
        h, w = img_padded.shape
        c = 1
        # 처리를 위해 (H, W, 1) 형태로 잠시 변환
        img_padded = img_padded[:, :, np.newaxis]

    low_freq_img = np.zeros_like(img_padded, dtype=np.float32)
    high_freq_img = np.zeros_like(img_padded, dtype=np.float32)

    # 채널별로 루프 수행
    for ch in range(c):
        channel_data = img_padded[:, :, ch].astype(np.float32)
        
        for i in range(0, h, block_size):
            for j in range(0, w, block_size):
                block = channel_data[i:i+block_size, j:j+block_size]
                
                # DCT 변환
                dct_block = cv2.dct(block)
                
                # 마스크 생성
                low_mask = np.zeros((block_size, block_size), dtype=np.float32)
                low_mask[:cutoff, :cutoff] = 1 
                
                # 저주파 성분 복원
                low_dct = dct_block * low_mask
                low_freq_img[i:i+block_size, j:j+block_size, ch] = cv2.idct(low_dct)
                
                # 고주파 성분 복원
                high_dct = dct_block * (1 - low_mask)
                high_freq_img[i:i+block_size, j:j+block_size, ch] = cv2.idct(high_dct)

    # 입력이 흑백(c=1)이었다면 다시 2D 배열로 되돌림
    if c == 1:
        low_freq_img = np.squeeze(low_freq_img)
        high_freq_img = np.squeeze(high_freq_img)
    else:
        # 시각화를 위해 uint8 변환이 필요할 수 있으므로 클리핑 처리
        low_freq_img = np.clip(low_freq_img, 0, 255).astype(np.uint8)
        high_freq_img = np.clip(high_freq_img, 0, 255).astype(np.uint8)
        # low_freq_img = cv2.cvtColor(low_freq_img, cv2.COLOR_YCrCb2BGR)
        # high_freq_img = cv2.cvtColor(high_freq_img, cv2.COLOR_YCrCb2BGR)

    return low_freq_img, high_freq_img