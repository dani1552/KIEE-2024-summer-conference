import matplotlib.pyplot as plt
import numpy as np
import json

with open('KIEE-2024-summer-conference/assets/020_00_0_0_nm.json', 'r') as f:
    data = json.load(f)

CONNECTIONS = [(0, 1), (0, 2), (0, 5), (0, 6), (1, 3), (2, 4), (5, 6), (5, 7), (5, 11), (6, 8), (6, 12), (7, 9), (8, 10), (11, 13), (12, 14), (13, 15), (14, 16)]

def draw_skeleton(keypoints, image_id):
    x_coordinates = np.array(keypoints[::2])
    y_coordinates = np.array(keypoints[1::2])

    # 관절 위치에 점을 찍기
    plt.plot(x_coordinates, y_coordinates, 'o', color='black')

    # 관절 위치 연결하는 선 그리기
    for connection in CONNECTIONS:
        plt.plot([x_coordinates[connection[0]], x_coordinates[connection[1]]],
                 [y_coordinates[connection[0]], y_coordinates[connection[1]]],
                 color='black')

    # 각 관절 위치에 번호 표시 및 console에 좌표 출력
    for i in range(len(x_coordinates)):
        plt.text(x_coordinates[i], y_coordinates[i], str(i), color='black', fontsize=10)
        print(f"Image ID: {image_id}, Keypoint {i}: [{x_coordinates[i]}, {y_coordinates[i]}]")

    plt.show()

# JSON 데이터 내의 모든 아이템에 대해 스켈레톤 그리기
for item in data:
    keypoints_data = item['keypoints']
    image_id = item['image_id']  
    keypoints = []

    # x, y 좌표만 추출하여 keypoints 리스트에 추가
    for i in range(0, len(keypoints_data), 3):
        x = keypoints_data[i]
        y = keypoints_data[i+1]
        keypoints.extend([x, y])

    # 추출한 키포인트로 스켈레톤을 그리기
    draw_skeleton(keypoints, image_id)  
