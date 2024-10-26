import cv2
import numpy as np

def count_non_black_pixels(image_path):
    """
    이미지에서 검은색(0, 0, 0)을 제외한 모든 픽셀을 1로 변환하고,
    해당하는 픽셀 수를 계산하는 함수입니다.
    
    Parameters:
        image_path (str): 이미지 파일 경로
        
    Returns:
        int: 검은색이 아닌 부분의 픽셀 수
    """
    # 이미지 불러오기
    image = cv2.imread(image_path)
    
    # 검은색과 일치하는 픽셀은 0, 나머지는 1로 변환
    binary_mask = np.where(np.all(image == [0, 0, 0], axis=-1), 0, 1)
    
    # 검은색이 아닌 부분의 픽셀 수 계산
    non_black_pixels = np.sum(binary_mask)
    
    return non_black_pixels

# 사용 예시
image_path = './raw_beef_json/label.png'  # 마스킹된 이미지 경로
non_black_pixel_count = count_non_black_pixels(image_path)
print(f'검은색이 아닌 픽셀 수: {non_black_pixel_count}')