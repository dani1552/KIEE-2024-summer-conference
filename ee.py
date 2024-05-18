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
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout,Conv3D, MaxPooling3D

gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
  try:
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)
  except RuntimeError as e:
    # 프로그램 시작시에 메모리 증가가 설정되어야만 합니다
    print(e)



def get_joint_point(data, bpaq):
    all_joint_img =[]
    joint_images_1 = []
    joint_images_2 = []
    joint_images_3 = []
    all_bpaq = [] 
    bpaq_data_1 = []
    bpaq_data_2 = []
    bpaq_data_3 = []

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
     
     
        # all_joint_img.append(img) # all_joint_img 리스트 업데이트
        img  = draw_pose(keypoints)
    
        if len(all_joint_img) < 15:
            joint_images_1.append(img)
            bpaq_data_1.append(bpaq)

        elif len(all_joint_img) < 30:
            joint_images_2.append(img)
            bpaq_data_1.append(bpaq)

        elif len(all_joint_img) < 45:
            joint_images_3.append(img)
            bpaq_data_3.append(bpaq)

    return (joint_images_1, bpaq_data_1), (joint_images_2, bpaq_data_2), (joint_images_3, bpaq_data_3)
    # return np.array(joint_imges_1, joint_imges_2, joint_imges_3)


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

def get_BPAQ():
    # CSV 파일 경로
    file_path = 'BPAQ_Hostility_Label.csv'

    # CSV 파일 읽기
    df = pd.read_csv(file_path)
    bpaq_data =  df["BPAQ_Hostility_Label"].to_list() 
    # print(len(bpaq_data))
    # # 'BPAQ'가 포함된 열 이름 찾기
    # bpaq_columns = [col for col in data.columns if 'BPAQ' in col]

    # # 'BPAQ'가 포함된 데이터만 가져오기
    # bpaq_data = data[bpaq_columns]

    # # 결과 출력
    # print(bpaq_data)
    # bpaq_physical_aggression = data['BPAQ_total'].tolist()
    
    # 0: Low , 1: Normal / Low, 2: Normal / High, 3: High

    new_bpaq_data = []


    for index in range(len(bpaq_data)):
        if(bpaq_data[index] =='Low'):
            new_bpaq_data.append(0)
        elif(bpaq_data[index] =='Normal / Low'):
            new_bpaq_data.append(1)
        elif(bpaq_data[index] =='Normal / High'):
            new_bpaq_data.append(2)
        elif(bpaq_data[index] =='High'):
            new_bpaq_data.append(3)

    return np.array(new_bpaq_data)


if __name__ == '__main__':
    all_joint_img_batches = []
    all_bpaq_data_batches = []
    all_joint_img =[]
    bpaq = get_BPAQ()
    print("Converting image...")
    for sub_id in range(312):
        pre = "00"
       
        if sub_id>=100:
            pre =""
        elif sub_id>=10:
            pre = "0"
        for j in range(1): #0번, 1번
            with open('KIEE-2024-summer-conference\\\data\\skeletons\\'+str(sub_id)+'\\'+pre+str(sub_id)+'_90_'+str(j)+'_0_nm.json', 'r') as f:
                data = json.load(f)
                bpaq = get_BPAQ() # bpaq 데이터를 불러오는 위치를 수정
                joint_img = get_joint_point(data) 
                joint_img_batch_1, joint_img_batch_2, joint_img_batch_3 = get_joint_point(data, bpaq)
                all_joint_img_batches.append(joint_img)
                all_joint_img_batches.extend([joint_img_batch_1, joint_img_batch_2, joint_img_batch_3]) # 이미지 배치 추가

    print("Complete onverting image...")
    new_joint_data = []
    new_bpaq = []

    for index, joint_img_batch in enumerate(all_joint_img_batches):
        if len(joint_img_batch) == 15:
            new_joint_data.append(np.array([joint_img_batch]))
            new_bpaq.append(np.array([bpaq[index]]))

    joint_data = new_joint_data
    bpaq = new_bpaq
    print(len(joint_data))

    input = np.array(joint_data)
    output =  np.array(bpaq)

    print(input.shape)
    print(output.shape)

    input = input.reshape(-1,15,100,100,1)

    print(input.shape)
    print(output.shape)

    X_train, X_test, y_train, y_test = train_test_split(input, output, test_size=0.2, shuffle= True)

    # 분할 후 데이터 길이 확인
    print(f"X_train 길이: {len(X_train)}, y_train 길이: {len(y_train)}")
    print(f"X_test 길이: {len(X_test)}, y_test 길이: {len(y_test)}")
    
    def check_data_consistency(X, y):
        if len(X) != len(y):
            raise ValueError(f"X와 y의 길이가 일치하지 않습니다. X의 길이: {len(X)}, y의 길이: {len(y)}")
        else:
            print("X와 y의 길이가 일치합니다.")

    # 훈련 데이터와 테스트 데이터에 대해 길이 확인
    check_data_consistency(X_train, y_train)
    check_data_consistency(X_test, y_test)


    model = Sequential()
    model.add(Conv3D(filters=64, kernel_size=(8,8,8), padding='same', activation='relu', input_shape = (15,100,100,1))) #입력데이터 한줄에 125개, 총 558줄
    model.add(MaxPooling3D(pool_size=(2, 2, 2)))
 
    model.add(Conv3D(filters=128, kernel_size=(4,4,4), padding='same', activation='relu'))
    model.add(MaxPooling3D(pool_size=(2, 2, 2)))
    
    model.add(Conv3D(filters=256, kernel_size=(2,2,2), padding='same', activation='relu'))
    model.add(MaxPooling3D(pool_size=(2, 2, 2)))
    model.add(Flatten())
    model.add(Dense(512, activation='relu'))
    model.add(Dense(512, activation='relu'))
    model.add(Dense(4, activation='softmax'))

    # model = ResNet152(input_shape=(50, 34, 1), include_top=False, weights=None)

    # 모델 컴파일
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

    # 모델 훈련
    model.fit(X_train, y_train, epochs=100, batch_size=16)  # 에포크 수 및 배치 크기는 상황에 맞게 조정하세요.
   
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
    print(f"Batch {index+1} Accuracy: {accuracy}")

    