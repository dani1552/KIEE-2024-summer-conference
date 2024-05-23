import csv
import json
import os
import cv2
import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import numpy as np
import json

import pandas as pdv
from sklearn.metrics import mean_squared_error,r2_score, accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, mean_absolute_error

from sklearn.model_selection import train_test_split
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout,Conv3D, MaxPooling3D
from keras.callbacks import EarlyStopping, LearningRateScheduler
from keras.optimizers import Adam
# TensorFlow가 GPU를 사용하는지 확인
print("using gpus? :", tf.test.is_built_with_cuda())

gpus = tf.config.experimental.list_physical_devices('GPU')
print("Num GPUs Available: ", len(gpus))

# gpus들 출력
for gpu in gpus:
    print("Name:", gpu.name, "Type:", gpu.device_type)

# GPU 메모리 사용량 증가 옵션 설정
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
  try:
    # 사용할 GPU 0,1 선택해서 메모리 증가 옵션 설정
    tf.config.experimental.set_memory_growth(gpus[0], True)
    tf.config.experimental.set_memory_growth(gpus[1], True)
  except RuntimeError as e:
    print(e)



def get_joint_point(data):
    global cnt
    all_joint_img =[]
    joint_images_1 = []
    joint_images_2 = []
    joint_images_3 = []
    joint_images_4 = []


    for item in data:
        keypoints_data = item['keypoints']
        image_id = item['image_id']  
        keypoints = []
        
        # 한 프레임의 x, y 좌표만 추출하여 keypoints 리스트에 추가
        for i in range(0, len(keypoints_data), 3):
            x = keypoints_data[i]
            y = keypoints_data[i+1]
            keypoints.extend([x, y]) #34개
    
        img = draw_pose(keypoints)

        if len(all_joint_img) < 15:
            joint_images_1.append(img)
            all_joint_img.append(img)  
            # print("joint_images_1: ", len(joint_images_1))
        elif len(all_joint_img) < 30:
            joint_images_2.append(img)
            all_joint_img.append(img) 
            # print("joint_images_2: ", len(joint_images_2))
        elif len(all_joint_img) < 45:
            joint_images_3.append(img)
            all_joint_img.append(img) 
            # print("joint_images_3: ", len(joint_images_3))
  

#    / print("joint images1,2,3 length before return: ", len(joint_images_1), len(joint_images_2), len(joint_images_3))
    return joint_images_1, joint_images_2, joint_images_3, joint_images_4

    # return np.array(joint_imges_1, joint_imges_2, joint_imges_3)





def draw_pose(keypoints):
    
    # COCO Pose Dataset의 관절 연결 순서
    CONNECTIONS = [(0, 1), (0, 2), (0, 5), (0, 6), (1, 3), (2, 4), (5, 6), (5, 7), (5, 11), (6, 8), (6, 12), (7, 9), (8, 10), (11, 13),  (11, 12), (12, 14), (13, 15), (14, 16)]
    
    # 좌표 추출
    x = keypoints[::2]
    y = keypoints[1::2]

    #이미지 속성 설정
    joint_radius=3
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
    img_array = img_result.astype('float16')
    # 변환된 이미지 확인
    # print(image_array)
    # # 변환된 이미지의 가로 세로 길이 확인
    # height, width, channels = image_array.shape
    # print("가로 길이:", width)
    # print("세로 길이:", height)
    
    return img_array

def get_gender():
    # CSV 파일 경로
    file_path = 'data\\metadata_raw_scores_v3.csv'

    # CSV 파일 읽기
    df = pd.read_csv(file_path)
    bpag_data =  df["ATTR_Gender"].to_list()

    new_bpag_data = []
    for index in range(len(bpag_data)):
        if(bpag_data[index] =='M'):
            new_bpag_data.append(0)
        elif(bpag_data[index] =='F'):
            new_bpag_data.append(1)
  

    return new_bpag_data



def result_classification(y_test, y_pred,history):
      
    y_pred = np.argmax(y_pred, axis=1)
    
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

    # 훈련 손실과 검증 손실을 시각화
    import matplotlib.pyplot as plt

    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.legend()
    plt.show()

    


    
def result_prediction(y_test, y_pred,history,save_filepath,):
      
    y_pred = y_pred.squeeze()

    f =  open(save_filepath, 'w', encoding='utf-8', newline="")
    wr = csv.writer(f)   
    wr.writerow(["Joint_Angle", "Predict_Angle"])
    for i in range(len(y_test)-1):
        wr.writerow([y_test[i][0], y_pred[i]])

    print("========= Result =========")
    print("MAE: ",mean_absolute_error(y_true=y_test, y_pred=y_pred))
    print("MAE2: ", np.mean(np.abs(y_test - y_pred)))
    print("std: ", np.std(np.absolute(np.subtract(y_test, y_pred))))
    print("=========================")
    print("ME: ", np.mean(np.subtract(y_test, y_pred)))
    print("std: ", np.std(np.subtract(y_test, y_pred)))
    print("=========================")
    print("mean relative error: ", np.mean(np.divide(np.absolute(np.subtract(y_test, y_pred)), y_test))*100)
    print("=========================")        
    print("r2 score: ", r2_score(y_true=y_test, y_pred=y_pred))
    print("=========================")
    
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.legend()
    plt.show()


# 학습률 스케줄링 함수 정의
# def lr_scheduler(epoch, lr):
#         decay_rate = 0.9883
#         decay_epoch = 1
#         if epoch % decay_epoch == 0 and epoch:
#             return lr * decay_rate
#         return lr

def VGG16():
    # Sequential 모델 선언
    model = keras.Sequential()
    # TODO : 지시사항을 잘 보고 VGG16 Net을 완성해보세요.
    
    # 첫 번째 Conv Block
    # 입력 Shape는 ImageNet 데이터 세트의 크기와 같은 RGB 영상 (224 x 224 x 3)입니다.
    model.add(keras.layers.Conv3D(filters = 64, kernel_size = (3,3,3), padding = "same" ,activation="relu", input_shape = (15,100,100,1)))
    model.add(keras.layers.Conv3D(filters = 64, kernel_size = (3,3,3), padding = "same", activation = "relu", ))
    model.add(keras.layers.MaxPooling3D((1,2,2)))
    
    # 두 번째 Conv Block
    model.add(keras.layers.Conv3D(filters = 128, kernel_size = (3,3,3), activation = "relu", padding = "same" ))
    model.add(keras.layers.Conv3D(filters = 128, kernel_size = (3,3,3), activation = "relu", padding = "same" ))
    model.add(keras.layers.MaxPooling3D((1,2,2))),
    
    # 세 번째 Conv Block
    model.add(keras.layers.Conv3D(filters = 256, kernel_size = (3,3,3), activation = "relu", padding = "same" ))
    model.add(keras.layers.Conv3D(filters = 256, kernel_size = (3,3,3), activation = "relu", padding = "same" ))
    model.add(keras.layers.Conv3D(filters = 256, kernel_size = (3,3,3), activation = "relu", padding = "same" ))
    model.add(keras.layers.MaxPooling3D((1,2,2))),
    
    # 네 번째 Conv Block
    model.add(keras.layers.Conv3D(filters = 512, kernel_size = (3,3,3), activation = "relu", padding = "same" ))
    model.add(keras.layers.Conv3D(filters = 512, kernel_size = (3,3,3), activation = "relu", padding = "same" ))
    model.add(keras.layers.Conv3D(filters = 512, kernel_size = (3,3,3), activation = "relu", padding = "same" ))
    model.add(keras.layers.MaxPooling3D((1,2,2))),

    
    # Fully Connected Layer
    model.add(keras.layers.Flatten()),
    model.add(keras.layers.Dense(1024, activation = "relu")),
    model.add(keras.layers.Dense(1024, activation = "relu")),
    model.add(keras.layers.Dense(2, activation = "softmax")) # 마지막 레이어의 노드수를 2로 변경하세요
    
    return model


if __name__ == '__main__':
    with tf.device('GPU:1') :
        all_joint_img_batches = []
        all_bpag_batches = []
        all_joint_img =[]
        bpag = get_gender()
        print(bpag)
        types = ['nm','ph']

        print("Converting image...")

        for sub_id in range(312):
            for type in types:
                pre = "00"
            
                if sub_id>=100:
                    pre =""
                elif sub_id>=10:
                    pre = "0"
                for j in range(1): #0번, 1번
                    with open('data\\skeletons\\'+str(sub_id)+'\\'+pre+str(sub_id)+'_90_'+str(j)+'_0_'+ type+'.json', 'r') as f:
                        data = json.load(f)
                        joint_images_1, joint_images_2, joint_images_3,joint_images_4 = get_joint_point(data)
                        all_joint_img_batches.append(joint_images_1)
                        all_joint_img_batches.append(joint_images_2)
                        all_joint_img_batches.append(joint_images_3)
                        all_bpag_batches.append(bpag[sub_id])
                        all_bpag_batches.append(bpag[sub_id])
                        all_bpag_batches.append(bpag[sub_id])

        print(len(all_joint_img_batches))
                        
        print("Complete Converting image...")
        new_joint_data = []
        new_bpag = []

        print(len(all_joint_img_batches))
        print(len(all_bpag_batches))

        for index, joint_img_batch in enumerate(all_joint_img_batches):
            if len(joint_img_batch) == 15:
                new_joint_data.append(np.array([joint_img_batch]))
                new_bpag.append(np.array([all_bpag_batches[index]]))

        joint_data = new_joint_data
        bpag = new_bpag
        print("(len(joint_data): ", len(joint_data))

        input = np.array(joint_data)
        output =  np.array(bpag)

        print("input.shape: ", input.shape)
        print("output.shape: ", output.shape)

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

        # 얼리 스타핑 콜백 설정
        early_stopping_callback = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=5)



        model = (VGG16)
        # model = ResNet152(input_shape=(50, 34, 1), include_top=False, weights=None)

        # 옵티마이저 설정
        optimizer = Adam(learning_rate=0.01)
        # 모델 컴파일
        model.compile(optimizer=optimizer, loss='sparse_categorical_crossentropy', metrics=['accuracy'])
        # model.compile(optimizer='adam', loss='mean_squared_error', metrics=['mae'])
        # lr_scheduler = LearningRateScheduler(lr_scheduler, verbose=1)

        # 모델 훈련
        history = model.fit(X_train, y_train, epochs=100,validation_split= 0.2, batch_size=32, shuffle=True,)  
    
 
        # 모델 예측
        y_pred = model.predict(X_test)
        
        result_classification(y_test ,y_pred, history)
        # result_prediction(y_test ,y_pred, history, 'predict_result_gait_cycle.csv')




        