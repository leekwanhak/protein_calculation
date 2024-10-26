import cv2
import numpy as np

def calculate_masked_pixels(image_path):
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
image_path_first = './raw_beef_json/label.png'  # 마스킹된 이미지 파일 경로
image_path_second = './fat.png'  # 마스킹된 이미지 파일 경로
masked_pixels_first = calculate_masked_pixels(image_path_first)
masked_pixels_second = calculate_masked_pixels(image_path_second)
print(f'첫번째 U-net 마스킹된 픽셀 수: {masked_pixels_first}')
print(f'두번째 U-net 마스킹된 픽셀 수: {masked_pixels_second}')


def calculate_pixels_per_cm2(reference_image_path):
    """
    스케일 미터 마커가 포함된 이미지를 사용하여 1cm²당 픽셀 수를 계산합니다.
    """
    # 참조 객체가 있는 이미지를 불러오기 (그레이스케일)
    image = cv2.imread(reference_image_path, cv2.IMREAD_GRAYSCALE)
    
    # Canny Edge Detector를 사용하여 윤곽 추출
    edges = cv2.Canny(image, 100, 200)
    
    # 윤곽선 검출을 통해 참조 객체(눈금자) 추출
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    reference_area_pixels = 0
    for contour in contours:
        # 눈금자 직사각형의 윤곽을 찾고 해당 픽셀 면적을 계산
        area = cv2.contourArea(contour)
        reference_area_pixels += area
    
    # 참조 객체의 실제 크기: 25 cm²
    reference_area_cm2 = 25
    
    # 1cm²당 픽셀 수 계산
    pixels_per_cm2 = reference_area_pixels / reference_area_cm2
    return pixels_per_cm2

def calculate_fat_percentage(fat_mask_pixels, beef_mask_pixels):
    """
    지방의 비율을 계산합니다.
    """
    fat_percentage = (fat_mask_pixels / beef_mask_pixels) * 100
    return fat_percentage

def calculate_fat_mass(fat_area_cm2, thickness_cm=3.81, fat_density_g_per_cm3=0.9):
    """
    지방의 질량을 계산합니다.
    지방의 면적, 두께, 밀도를 곱하여 계산.
    """
    fat_mass = fat_area_cm2 * thickness_cm * fat_density_g_per_cm3
    return fat_mass

# 1. 참조 이미지 경로
reference_image_path = './raw_beef.png'  # 스케일 미터 마커가 있는 이미지

# 2. 1cm²당 픽셀 수 계산
pixels_per_cm2 = calculate_pixels_per_cm2(reference_image_path)
print(f'1cm²당 픽셀 수: {pixels_per_cm2:.2f}')

# 3. 지방 마스크 픽셀 수와 소고기 마스크 픽셀 수 (예시 값)
fat_mask_pixels = masked_pixels_second  # 두 번째 U-Net에서 얻은 지방 마스크 픽셀 수
beef_mask_pixels = masked_pixels_first  # 첫 번째 U-Net에서 얻은 소고기 마스크 픽셀 수

# 4. 지방 비율 계산
fat_percentage = calculate_fat_percentage(fat_mask_pixels, beef_mask_pixels)
print(f'지방 비율: {fat_percentage:.2f}%')

# 5. 지방 면적 계산 (cm²)
fat_area_cm2 = fat_mask_pixels / pixels_per_cm2

# 6. 지방 질량 계산 (g)
fat_mass = calculate_fat_mass(fat_area_cm2)
print(f'지방 질량: {fat_mass:.2f}g')
