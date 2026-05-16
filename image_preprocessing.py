import os
import cv2
import numpy as np
from datasets import load_dataset

# 저장 폴더 생성
output_dir = 'preprocessed_samples'
os.makedirs(output_dir, exist_ok=True)

# 데이터셋 로드
dataset = load_dataset('ethz/food101', split='train', streaming=True)

saved_count = 0

for i, sample in enumerate(dataset):
    if saved_count >= 5:  # 정확히 총 5장이 채워지면 종료
        break
        
    # 이미지 가져오고 포맷 변환
    pil_image = sample['image']
    open_cv_image = np.array(pil_image)
    
    if len(open_cv_image.shape) == 3:
        image = cv2.cvtColor(open_cv_image, cv2.COLOR_RGB2BGR)
    else:
        continue 

    # [심화 문제] 이상치 탐지 및 필터링 알고리즘
    gray_check = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # 1) 너무 어두운 이미지 제거 
    avg_brightness = np.mean(gray_check)
    if avg_brightness < 50:
        continue

    # 2) 객체 크기가 너무 작은 이미지 제거
    _, thresh = cv2.threshold(gray_check, 127, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if contours:
        max_area = max([cv2.contourArea(c) for c in contours])
        if max_area < 1500:
            continue
    else:
        continue

    # [기본 문제] 
    # 크기 조정 (224 x 224)
    resized = cv2.resize(image, (224, 224))
    
    # 색상 변환
    gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
    
    # 정규화
    normalized = gray / 255.0
    
    # 노이즈 제거
    blur = cv2.GaussianBlur((normalized * 255).astype(np.uint8), (5, 5), 0)

    # 데이터 증강
    flipped = cv2.flip(blur, 1)  # 좌우 반전
    rotated = cv2.rotate(flipped, cv2.ROTATE_90_CLOCKWISE)  # 90도 회전
    final_processed = cv2.convertScaleAbs(rotated, alpha=1.0, beta=30)  # 밝기(색상) 변화

    # 최종 5장
    cv2.imwrite(f"{output_dir}/processed_img_{saved_count + 1}.jpg", final_processed)
    
    saved_count += 1