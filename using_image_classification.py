import json
import os
import cv2
import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import numpy as np
import json

import pandas as pd
from sklearn.metrics import mean_squared_error,r2_score, accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, mean_absolute_error

from sklearn.model_selection import train_test_split
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout

gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
  try:
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)
  except RuntimeError as e:
    # 프로그램 시작시에 메모리 증가가 설정되어야만 합니다
    print(e)

def get_joint_point(data):
    all_joint_img =[]

    # 한 사람에 대한 데이터 
    for item in data:
        keypoints_data = item['keypoints']
        image_id = item['image_id']  
        keypoints = []

        # 한 프레임의 x, y 좌표만 추출하여 keypoints 리스트에 추가
        for i in range(0, len(keypoints_data), 3):
            x = keypoints_data[i]
            y = keypoints_data[i+1]
            keypoints.extend([x, y]) #34개
        print
        img  = draw_pose(keypoints)
        print(sum(img[20]))
        exit()

        if(len(all_joint_img)<50):       
            all_joint_img.append(img)

    return np.array(all_joint_img)


def draw_pose(keypoints):
    
    # COCO Pose Dataset의 관절 연결 순서
    CONNECTIONS = [(0, 1), (0, 2), (0, 5), (0, 6), (1, 3), (2, 4), (5, 6), (5, 7), (5, 11), (6, 8), (6, 12), (7, 9), (8, 10), (11, 13), (12, 14), (13, 15), (14, 16)]
    
    # 좌표 추출
    x = keypoints[::2]
    y = keypoints[1::2]

    #이미지 속성 설정
    joint_radius=4
    joint_color =(0, 0, 0)
    line_color =(0, 0, 0)
    line_thickness = 2
    image_size=(100, 100)

    # 좌표의 최대 및 최소값 계산
    min_x, min_y = min(x), min(y)
    max_x, max_y = max(x), max(y)

    # 이미지를 그릴 영역 계산
    img_width = max_x - min_x
    img_height = max_y - min_y
    scale_factor = min((image_size[0] - 20) / img_width, (image_size[1] - 20) / img_height)
    new_width = int(img_width * scale_factor)
    new_height = int(img_height * scale_factor)
    offset_x = (image_size[0] - new_width) // 2
    offset_y = (image_size[1] - new_height) // 2

    # 이미지 생성
    img = np.ones((image_size[1], image_size[0], 3), dtype=np.uint8) * 255

    # 관절 위치에 원 그리기
    for joint_x, joint_y in zip(x, y):
        x_pixel = int((joint_x - min_x) * scale_factor) + offset_x
        y_pixel = int((joint_y - min_y) * scale_factor) + offset_y
        cv2.circle(img, (x_pixel, y_pixel), joint_radius, joint_color, -1)

    # 관절 위치 연결하는 선 그리기
    for connection in CONNECTIONS:
        joint1_x, joint1_y = x[connection[0]], y[connection[0]]
        joint2_x, joint2_y = x[connection[1]], y[connection[1]]
        x1_pixel = int((joint1_x - min_x) * scale_factor) + offset_x
        y1_pixel = int((joint1_y - min_y) * scale_factor) + offset_y
        x2_pixel = int((joint2_x - min_x) * scale_factor) + offset_x
        y2_pixel = int((joint2_y - min_y) * scale_factor) + offset_y
        cv2.line(img, (x1_pixel, y1_pixel), (x2_pixel, y2_pixel), line_color, line_thickness)


    # 생성된 이미지를 파일로 저장하거나 화면에 표시할 수 있습니다.
    # cv2.imwrite('pose_image.jpg', img)
    # cv2.imshow('Pose Image', img)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    # OpenCV 이미지를 ndarray로 변환

    gray_image = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img_result = gray_image / 255.0

    # 변환된 이미지 확인
    # print(image_array)
    # # 변환된 이미지의 가로 세로 길이 확인
    # height, width, channels = image_array.shape
    # print("가로 길이:", width)
    # print("세로 길이:", height)
    
    return img_result



def get_raw_BPAG():
    # CSV 파일 경로
    file_path = 'data\\metadata_raw_scores_v3.csv'

    # CSV 파일 읽기
    data = pd.read_csv(file_path)
    bpaq_physical_aggression = data['BPAQ_total'].tolist()
 
    return np.array(bpaq_physical_aggression)

def get_BPAG():
    # CSV 파일 경로
    file_path = 'classification_aggression\\BPAQ_Anger_Label.csv'

    # CSV 파일 읽기
    df = pd.read_csv(file_path)
    bpag_data =  df["BPAQ_Anger_Label"].to_list()
    print(len(bpag_data))
    # # 'BPAQ'가 포함된 열 이름 찾기
    # bpaq_columns = [col for col in data.columns if 'BPAQ' in col]

    # # 'BPAQ'가 포함된 데이터만 가져오기
    # bpaq_data = data[bpaq_columns]

    # # 결과 출력
    # print(bpaq_data)
    # bpaq_physical_aggression = data['BPAQ_total'].tolist()
    
    #0: Low , 1: Normal / Low, 2: Normal / High, 3: High

    new_bpag_data = []
    for index in range(len(bpag_data)):
        if(bpag_data[index] =='Low'):
            new_bpag_data.append(0)
        elif(bpag_data[index] =='Normal / Low'):
            new_bpag_data.append(1)
        elif(bpag_data[index] =='Normal / High'):
            new_bpag_data.append(2)
        elif(bpag_data[index] =='High'):
            new_bpag_data.append(3)

    return np.array(new_bpag_data)


if __name__ == '__main__':
    all_joint_img =[]
    bpag = get_BPAG()
    print("Converting image...")
    for sub_id in range(312): #사람 50명 예시
        pre = "00"
       
        if sub_id>=100:
            pre =""
        elif sub_id>=10:
            pre = "0"
        for j in range(1): #0번, 1번
            with open('data\\skeletons\\'+str(sub_id)+'\\'+pre+str(sub_id)+'_45_'+str(j)+'_0_nm.json', 'r') as f:
                data = json.load(f)
                joint_img = get_joint_point(data)
                all_joint_img.append(joint_img)

    print("Complete onverting image...")
    new_joint_data = []
    new_bpag = []

    for index in range(len(all_joint_img)):
        if len(all_joint_img[index]) == 50:
            new_joint_data.append(all_joint_img[index])
            new_bpag.append(bpag[index])

    joint_data = new_joint_data
    bpag = new_bpag

    input = np.array(joint_data)
    output =  np.array(bpag)

    print(input.shape)
    print(output.shape)

    input = input.reshape(-1,50,100,100)
    output.squeeze()

    print(input.shape)
    print(output.shape)

    X_train, X_test, y_train, y_test = train_test_split(input, output, test_size=0.2, shuffle= True)

    
    model = Sequential()
    model.add(Conv2D(64, (3, 3), activation='relu', input_shape=(50, 100, 100)))
    model.add(MaxPooling2D((2, 2)))
    # Dropout 레이어 추가 (0.25는 비활성화할 뉴런의 비율)
    model.add(Dropout(0.25))
    model.add(Conv2D(128, (3, 3), activation='relu'))
    model.add(MaxPooling2D((2, 2)))
    # Dropout 레이어 추가 (0.25는 비활성화할 뉴런의 비율)
    model.add(Dropout(0.25))
    model.add(Conv2D(256, (3, 3), activation='relu'))
    model.add(MaxPooling2D((2, 2)))
    model.add(Flatten())
    model.add(Dense(512, activation='relu'))
    model.add(Dense(512, activation='relu'))
    model.add(Dense(512, activation='relu'))
    model.add(Dense(512, activation='relu'))
    model.add(Dense(4, activation='softmax'))

    # model = ResNet152(input_shape=(50, 34, 1), include_top=False, weights=None)

    # 모델 컴파일
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

    # 모델 훈련
    model.fit(X_train, y_train, epochs=5000, batch_size=64)  # 에포크 수 및 배치 크기는 상황에 맞게 조정하세요.
   
    # 모델 예측
    y_pred_prob = model.predict(X_test)
  
    y_pred = np.argmax(y_pred_prob, axis=1)
    
    # 평가 지표 계산
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average='weighted')
    recall = recall_score(y_test, y_pred, average='weighted')
    f1 = f1_score(y_test, y_pred, average='weighted')
    conf_matrix = confusion_matrix(y_test, y_pred)

    # 결과 출력
    print("정확도:", accuracy)
    print("정밀도:", precision)
    print("재현율:", recall)
    print("F1 점수:", f1)
    print("혼동 행렬:\n", conf_matrix)

    