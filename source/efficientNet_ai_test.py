import pandas as pd
from sklearn import metrics
import os
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import ImageDataGenerator

os.environ["CUDA_VISIBLE_DEVICES"]="0"

modelPath = '/home/mark11/label/model_saved/핵경화/efficientnet_200/'  # 모델이 저장된 경로
weight = 'model-130-0.997294-0.997083.h5'        # 학습된 모델의 파일이름
test_Path = '/home/mark11/label/dog/Sclerosis/test/' # 테스트 이미지 폴더

model = load_model(modelPath + weight)
# model = load_model(modelPath)
# datagen_test = ImageDataGenerator(rescale=1./255)
datagen_test = ImageDataGenerator()
generator_test = datagen_test.flow_from_directory(directory=test_Path,
                                                  target_size=(224, 224),
                                                  batch_size=256,
                                                  shuffle=False)

# model로 test set 추론
generator_test.reset()
cls_test = generator_test.classes
cls_pred = model.predict_generator(generator_test, verbose=1, workers=0)
cls_pred_argmax = cls_pred.argmax(axis=1)

# 결과 산출 및 저장
report = metrics.classification_report(y_true=cls_test, y_pred=cls_pred_argmax, output_dict=True)
report = pd.DataFrame(report).transpose()
report.to_csv(f'./output/report_test_{weight[:-3]}.csv', index=True, encoding='cp949')
print(report)
