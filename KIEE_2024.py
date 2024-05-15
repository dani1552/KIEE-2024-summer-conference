import matplotlib.pyplot as plt
import numpy as np
import json

with open('assets/020_00_0_0_nm.json', 'r') as f:
    data = json.load(f)

# 관절 연결 정보
CONNECTIONS = [(0, 1), (0, 2), (0, 5), (0, 6), (1, 3), (2, 4), (5, 6), (5, 7), (5, 11), (6, 8), (6, 12), (7, 9), (8, 10), (11, 13), (12, 14), (13, 15), (14, 16)]

# keypoints를 그래프로 그리는 함수를 정의합니다.
def plot_keypoints(keypoints):

    # x좌표, y좌표 생성
    x_coordinates = np.array(keypoints[::2])
    y_coordinates = np.array(keypoints[1::2]) 

    # 관절 위치 표시
    plt.plot(x_coordinates, y_coordinates, 'o', color='black')
    
    # 모든 관절 연결
    for connection in CONNECTIONS:
        plt.plot([x_coordinates[connection[0]], x_coordinates[connection[1]]],
                 [y_coordinates[connection[0]], y_coordinates[connection[1]]], 
                 color='black')
        
    # 각 점에 번호 표시
    for i in range(len(x_coordinates)):
        plt.text(x_coordinates[i], y_coordinates[i], str(i), color='black', fontsize=10)
    
    plt.show()

# 모든 데이터 요소에 대해 'keypoints' 정보 가져와 그래프 그리기
for item in data:
    keypoints_data = item['keypoints']
    keypoints = []

    # 'keypoints_data'에서 'keypoints' 정보 추출 & 리스트에 추가
    for i in range(0, len(keypoints_data), 3):
        x = keypoints_data[i]
        y = keypoints_data[i+1]
        keypoints.extend([x, y])

    # 그래프 그리는 함수 호출
    plot_keypoints(keypoints)
