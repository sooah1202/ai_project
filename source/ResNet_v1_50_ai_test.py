import pandas as pd
from sklearn import metrics

from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# modelPath = './model_saved/train/efficientnet_200/'  # 모델이 저장된 경로
modelPath = '/home/mark11/Dog/model_saved/train/resnet_v1_50_100/'  # 모델이 저장된 경로
weight = 'model-066-0.874390-0.860614.h5'        # 학습된 모델의 파일이름
test_Path = '/home/mark11/label/cat/Sequestrum/test/' # 테스트 이미지 폴더

model = load_model(modelPath + weight) #모델을 가져옴
datagen_test = ImageDataGenerator(rescale=1./255)
generator_test = datagen_test.flow_from_directory(directory=test_Path,
                                                  target_size=(224, 224),
                                                  batch_size=256,
                                                  shuffle=False) #테스트 데이터를 가져와서 전처리 및 셋팅

# model로 test set 추론
generator_test.reset() #predict_generator 사용시 해줘야함
cls_test = generator_test.classes #label얻기
cls_pred = model.predict_generator(generator_test, verbose=1, workers=0) #모델 예측(데이터, 세부정보표시, 병렬 처리에 사용할 최대 스레드 수)
cls_pred_argmax = cls_pred.argmax(axis=1) #행의 최대값의 인덱스 위치를 배열로 가져옴

# 결과 산출 및 저장
report = metrics.classification_report(y_true=cls_test, y_pred=cls_pred_argmax, output_dict=True)
report = pd.DataFrame(report).transpose()
report.to_csv(f'./output/report_test_{weight[:-3]}.csv', index=True, encoding='cp949') #결과를 csv 파일로 저장
print(report)