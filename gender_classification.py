import csv
import json
import os
import random
import cv2
import numpy as np
import pandas as pd

import matplotlib.pyplot as plt

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report
from sklearn.model_selection import train_test_split
import tensorflow as tf
from keras.models import Model
from keras.layers import Input, Conv3D, MaxPooling3D, Flatten, Dense, Dropout, Add, BatchNormalization, Activation
from keras.callbacks import ReduceLROnPlateau
from imblearn.over_sampling import RandomOverSampler
from imblearn.over_sampling import SMOTE

# TensorFlow가 GPU를 사용하는지 확인
print("using gpus? :", tf.test.is_built_with_cuda())    # CUDA = GPU 하드웨어 지원해주는 플랫폼, tf.test.is_built_with_cuda()가 true 반환 -> CUDA 사용 가능!

gpus = tf.config.experimental.list_physical_devices('GPU')  # list_physical_devices('GPU'): TensorFlow가 현재 사용가능한 GPU 장치 목록 반환, 없으면 빈 목록 반환
print("Num GPUs Available: ", len(gpus))    # l환en(gpus): 시스템에 사용 가능한 GPU의 수 반환

# gpus들 출력
for gpu in gpus:
    print("Name:", gpu.name, "Type:", gpu.device_type)  # 장치의 이름, 

# GPU 메모리 사용량 증가 옵션 설정
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    except RuntimeError as e:
        print(e)

# JSON에서 관절좌표 추출 -> 포즈 이미지 생성
def get_joint_point(data):
    all_joint_img = []
    joint_images_1 = []
    joint_images_2 = []
    joint_images_3 = []
    joint_images_4 = []

    for item in data:
        keypoints_data = item['keypoints']
        keypoints = []

        for i in range(0, len(keypoints_data), 3):
            x = keypoints_data[i]
            y = keypoints_data[i+1]
            keypoints.extend([x, y])
    
        img = draw_pose(keypoints)

        if len(all_joint_img) < 15:
            joint_images_1.append(img)
            all_joint_img.append(img)  
        elif len(all_joint_img) < 30:
            joint_images_2.append(img)
            all_joint_img.append(img) 
        elif len(all_joint_img) < 45:
            joint_images_3.append(img)
            all_joint_img.append(img) 
            
    return joint_images_1, joint_images_2, joint_images_3, joint_images_4

# 관절 좌표를 받아 이미지로 변환, 각 관절을 원으로 그리고 관절 사이의 연결을 선으로 그림
def draw_pose(keypoints):
    CONNECTIONS = [(0, 1), (0, 2), (0, 5), (0, 6), (1, 3), (2, 4), (5, 6), (5, 7), (5, 11), (6, 8), (6, 12), (7, 9), (8, 10), (11, 13), (11, 12), (12, 14), (13, 15), (14, 16)]
    x = keypoints[::2]
    y = keypoints[1::2]

    joint_radius = 3
    joint_color = (0, 0, 0)
    line_color = (0, 0, 0)
    line_thickness = 2
    image_size = (50, 50)

    min_x, min_y = min(x), min(y)
    max_x, max_y = max(x), max(y)

    img_width = max_x - min_x
    img_height = max_y - min_y
    scale_factor = min((image_size[0] - 20) / img_width, (image_size[1] - 20) / img_height)
    new_width = int(img_width * scale_factor)
    new_height = int(img_height * scale_factor)
    offset_x = (image_size[0] - new_width) // 2
    offset_y = (image_size[1] - new_height) // 2

    img = np.ones((image_size[1], image_size[0], 3), dtype=np.uint8) * 255

    for joint_x, joint_y in zip(x, y):
        x_pixel = int((joint_x - min_x) * scale_factor) + offset_x
        y_pixel = int((joint_y - min_y) * scale_factor) + offset_y
        cv2.circle(img, (x_pixel, y_pixel), joint_radius, joint_color, -1)

    for connection in CONNECTIONS:
        joint1_x, joint1_y = x[connection[0]], y[connection[0]]
        joint2_x, joint2_y = x[connection[1]], y[connection[1]]
        x1_pixel = int((joint1_x - min_x) * scale_factor) + offset_x
        y1_pixel = int((joint1_y - min_y) * scale_factor) + offset_y
        x2_pixel = int((joint2_x - min_x) * scale_factor) + offset_x
        y2_pixel = int((joint2_y - min_y) * scale_factor) + offset_y
        cv2.line(img, (x1_pixel, y1_pixel), (x2_pixel, y2_pixel), line_color, line_thickness)

    gray_image = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img_result = gray_image / 255.0
    img_array = img_result.astype('float16')
    
    return img_array

# csv 파일에서 성별 데이터를 읽어옴 -> 'M' = 0, 'F' = 1으로 변환해서 리스트로 반환
def get_gender():
    file_path = 'workspace/data/metadata_raw_scores_v3.csv'
    df = pd.read_csv(file_path)
    gender_data = df["ATTR_Gender"].to_list()

    new_gender_data = []
    for index in range(len(gender_data)):
        if gender_data[index] == 'M':
            new_gender_data.append(0)
        elif gender_data[index] == 'F':
            new_gender_data.append(1)
  
    return new_gender_data

# 함수의 테스트 결과 평가, 다양한 지표 출력, 훈련 과정에서의 손실 및 정확도 곡선 저장
def result_classification(y_test, y_pred, history):
    y_pred_classes = (y_pred > 0.5).astype(int).flatten()
    
    accuracy = accuracy_score(y_test, y_pred_classes)
    precision = precision_score(y_test, y_pred_classes, average='weighted', zero_division=1)
    recall = recall_score(y_test, y_pred_classes, average='weighted')
    f1 = f1_score(y_test, y_pred_classes, average='weighted')
    conf_matrix = confusion_matrix(y_test, y_pred_classes)
    class_report = classification_report(y_test, y_pred_classes, zero_division=1)

    print("정확도:", accuracy)
    print("정밀도:", precision)
    print("재현율:", recall)
    print("F1 점수:", f1)
    print("혼동 행렬:\n", conf_matrix)
    print("분류 보고서:\n", class_report)

    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig('loss_curve.png')  # 파일로 저장

    plt.plot(history.history['accuracy'], label='Training Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.savefig('accuracy_curve.png')  # 파일로 저장

# ResNet3D 모델을 정의함, 'build_resnet_block': ResNet 블록 생성, 'build_model': 전체 모델 구성
class ResNet3D:
    def __init__(self, input_shape, num_classes):
        self.input_shape = input_shape
        self.num_classes = num_classes
        self.model = self.build_model()

    def build_resnet_block(self, input, filters, kernel_size, strides, use_bias=True):
        x = Conv3D(filters, kernel_size, strides=strides, padding='same', use_bias=use_bias)(input)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)
        x = Conv3D(filters, kernel_size, strides=1, padding='same', use_bias=use_bias)(x)
        x = BatchNormalization()(x)
        
        if input.shape[-1] != filters:
            input = Conv3D(filters, kernel_size=1, strides=strides, padding='same', use_bias=use_bias)(input)
        
        x = Add()([x, input])
        x = Activation('relu')(x)
        return x

    def build_model(self):
        input = Input(shape=self.input_shape)
        
        x = Conv3D(64, kernel_size=7, strides=2, padding='same', use_bias=False)(input)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)
        x = MaxPooling3D(pool_size=3, strides=2, padding='same')(x)
        
        for filters in [64, 128, 256, 512]:
            x = self.build_resnet_block(x, filters, kernel_size=3, strides=1)
            x = self.build_resnet_block(x, filters, kernel_size=3, strides=1)
        
        x = Flatten()(x)
        x = Dropout(0.5)(x)  # Dropout 추가
        x = Dense(512, activation='relu')(x)
        x = Dropout(0.5)(x)  # Dropout 추가
        x = Dense(self.num_classes, activation='sigmoid')(x)  # Sigmoid 활성화 함수 사용
        
        model = Model(inputs=input, outputs=x)
        return model

if __name__ == '__main__':
    all_joint_img_batches = []
    all_bpag_batches = []
    all_joint_img = []
    bpag = get_gender() # 성별 데이터를 가져옴

    types = ['nm', 'cl', 'bg', 'txt', 'ph', 'wss', 'wsf'] 
    
    print("Converting image...")

    for sub_id in range(156):
        for type in types:
            pre = "00"
            if sub_id>=100:
                pre =""
            elif sub_id>=10:
                pre = "0"
            for j in range(1): #0번, 1번
                with open('workspace/data/skeletons/'+str(sub_id)+'/'+pre+str(sub_id)+':90:'+str(j)+':0:'+ type+'.json', 'r') as f:
                    data = json.load(f)
                    joint_images_1, joint_images_2, joint_images_3,joint_images_4 = get_joint_point(data)
                    all_joint_img_batches.append(joint_images_1)
                    all_joint_img_batches.append(joint_images_2)
                    all_joint_img_batches.append(joint_images_3)
                    all_bpag_batches.append(bpag[sub_id])
                    all_bpag_batches.append(bpag[sub_id])
                    all_bpag_batches.append(bpag[sub_id])
            
    print("Complete Converting image...")
    new_joint_data = []
    new_bpag = []

    print(len(all_joint_img_batches))
    print(len(all_bpag_batches))

    for index, joint_img_batch in enumerate(all_joint_img_batches):
        if len(joint_img_batch) == 15:
            new_joint_data.append(np.array([joint_img_batch]))
            new_bpag.append(np.array(all_bpag_batches[index]))
    
    # 기존 코드
    joint_data = new_joint_data
    bpag = new_bpag

    input = np.array(joint_data)
    output = np.array(bpag)

    # reshape to 2D for SMOTE
    nsamples, nx, ny, nz, nchannel = input.shape
    input_reshaped = input.reshape((nsamples, nx * ny * nz * nchannel))

    # SMOTE oversampling
    smote = SMOTE(random_state=42)
    input_resampled, output_resampled = smote.fit_resample(input_reshaped, output)

    # reshape back to 3D
    input_resampled = input_resampled.reshape((len(input_resampled), nx, ny, nz, nchannel))

    input = input_resampled
    output = output_resampled

    print("input.shape: ", input.shape)
    print("output.shape: ", output.shape)

    input = input.reshape(-1, 15, 50, 50, 1)  # 이미지 크기를 50x50으로 변경

    print(input.shape)
    print(output.shape)

    X_train, X_test, y_train, y_test = train_test_split(input, output, test_size=0.1, shuffle=True)

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

    # 모델 빌드 및 훈련
    input_shape = (15, 50, 50, 1)
    num_classes = 1  # 단일 출력 뉴런
    resnet3d = ResNet3D(input_shape, num_classes)
    model = resnet3d.model

    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=20, min_lr=1e-5)

    history = model.fit(X_train, y_train, epochs=100, validation_split=0.2, callbacks=[reduce_lr])

    y_pred = model.predict(X_test)
    result_classification(y_test, y_pred, history)