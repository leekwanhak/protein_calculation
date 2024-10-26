import cv2
import numpy as np

def calculate_masked_pixels(image_path, mask_color=(255, 255, 255)):
    """
    마스킹된 이미지에서 지정된 색상의 픽셀 수를 계산합니다.
    
    Parameters:
        image_path (str): 이미지 파일 경로
        mask_color (tuple): 마스크의 RGB 색상 (기본값은 흰색)
        
    Returns:
        int: 마스킹된 픽셀 수
    """
    # 이미지 읽기
    image = cv2.imread(image_path)
    
    # 마스크 색상과 일치하는 픽셀을 찾아 마스크 생성
    mask = cv2.inRange(image, mask_color, mask_color)
    
    # 마스킹된 픽셀 수 계산
    masked_pixels = cv2.countNonZero(mask)
    
    return masked_pixels

# 사용 예시
image_path = './raw_beef_json/label.png'  # 마스킹된 이미지 파일 경로
masked_pixels = calculate_masked_pixels(image_path)
print(f'마스킹된 픽셀 수: {masked_pixels}')
