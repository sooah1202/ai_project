#-*- coding:utf-8 -*-
import matplotlib.pyplot as plt #모형 학습시 accuracy와 loss를 저장하기 위한 라이브러리입니다.

"""시드 고정을 위한 라이브러리"""
import random
import numpy as np

"""전처리를 위한 라이브러리"""
import os
import pandas as pd

"""Keras 라이브러리"""
#import tf.keras as keras #keras 라이브러리입니다.
import tensorflow.python.keras
import tensorflow as tf
from keras.preprocessing.image import ImageDataGenerator #이미지 데이터를 tensor로 변한하기 위해 활용되는 라이브러리입니다.
from keras.layers import Dense #학습 모형을 구축하기 위해 활용되는 라이브러리입니다.
from keras import Sequential #학습 모형을 구축하기 위해 활용되는 라이브러리 입니다.

from keras.applications.resnet import ResNet50, ResNet101, ResNet152
from keras.applications.resnet_v2 import ResNet50V2, ResNet101V2, ResNet152V2
from keras.applications.densenet import DenseNet121, DenseNet169, DenseNet201
from keras.applications.inception_v3 import InceptionV3
from keras.applications.inception_resnet_v2 import InceptionResNetV2
#from keras.applications.efficientnet import EfficientNetB0
# from tensorflow.keras.utils import multi_gpu_model
from keras.utils.multi_gpu_utils import multi_gpu_model
from keras.callbacks import ModelCheckpoint
import keras.backend as K
import keras as keras

val_path = '../data/label/cat/Sequestrum/val/'

seed = 2

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)

class Import_data:
    def __init__(self, train_path):
        self.train_path = train_path
        self.test_path = val_path

    def train(self):
        train_datagen = ImageDataGenerator(rescale=1. / 255, #데이터에 제공된 값을 곱하여 규모조정
                                           featurewise_std_normalization=True,#인풋을 각 특성 내에서 데이터셋의 표준편차로 나눔
                                           zoom_range=0.2, #플로트 또는 [아래, 위]. 임의 확대/축소 범위
                                           channel_shift_range=0.1,
                                           rotation_range=20, #무작위 회전의 각도 범위
                                           width_shift_range=0.2, #전체 가로넓이에서의 비율
                                           height_shift_range=0.2, #전체 세로높이에서의 비율
                                           horizontal_flip=True #입력을 임의로 수평으로 뒤집음
                                           ) #tensorflow's method(데이터 전처리)
        train_generator = train_datagen.flow_from_directory(
            self.train_path,
            target_size=(224, 224), #모든 이미지의 크기를 재조정할 치수
            batch_size=8 #데이터 배치의 크기(한번에 처리할 데이터의 수)
        ) #디렉토리에의 경로를 전달받아 증강된 데이터의 배치를 생성(train_path의 구조에 imageDateGenerator의 객체를 씌움)
        val_generator = train_datagen.flow_from_directory(
            self.test_path,
            target_size=(224, 224),
            batch_size=8
        )

        return train_generator, val_generator


def densenet_121():
    network = DenseNet121(include_top=False, weights='imagenet', input_tensor=None, input_shape=(224, 224, 3),
                          pooling='avg')
    return network


class Load_model:
    def __init__(self, train_path, model_name):
        self.num_class = len(os.listdir(train_path))
        self.model_name = model_name

    def resnet_v1_50(self):
        network = ResNet50(include_top=False, weights='imagenet', input_tensor=None, input_shape=(224, 224, 3),
                           pooling='avg')
        return network

    def resnet_v1_101(self):
        network = ResNet101(include_top=False, weights='imagenet', input_tensor=None, input_shape=(224, 224, 3),
                            pooling='avg')
        return network

    def resnet_v1_152(self):
        network = ResNet152(include_top=False, weights='imagenet', input_tensor=None, input_shape=(224, 224, 3),
                            pooling='avg')
        return network

    def resnet_v2_50(self):
        network = ResNet50V2(include_top=False, weights='imagenet', input_tensor=None, input_shape=(224, 224, 3),
                             pooling='avg')
        return network

    def resnet_v2_101(self):
        network = ResNet101V2(include_top=False, weights='imagenet', input_tensor=None, input_shape=(224, 224, 3),
                              pooling='avg')
        return network

    def resnet_v2_152(self):
        network = ResNet152V2(include_top=False, weights='imagenet', input_tensor=None, input_shape=(224, 224, 3),
                              pooling='avg')
        return network

    def densenet_169(self):
        network = DenseNet169(include_top=False, weights='imagenet', input_tensor=None, input_shape=(224, 224, 3),
                              pooling='avg')
        return network

    def densenet_201(self):
        network = DenseNet201(include_top=False, weights='imagenet', input_tensor=None, input_shape=(224, 224, 3),
                              pooling='avg')
        return network

    def inception_v3(self):
        network = InceptionV3(include_top=False, weights='imagenet', input_tensor=None, input_shape=(224, 224, 3),
                              pooling='avg')
        return network

    def inception_v4(self):
        network = InceptionResNetV2(include_top=False, weights='imagenet', input_tensor=None, input_shape=(224, 224, 3),
                                    pooling='avg')
        return network

    #def efficientnet(self):
        #network = EfficientNetB0(include_top=False, weights='imagenet', input_tensor=None, input_shape=(224, 224, 3),
                                 #pooling='avg')
        #return network


    def build_network(self):
        if self.model_name == 'resnet_v1_50':
            network = self.resnet_v1_50()
        elif self.model_name == 'resnet_v1_101':
            network = self.resnet_v1_101()
        elif self.model_name == 'resnet_v1_152':
            network = self.resnet_v1_152()
        elif self.model_name == 'resnet_v2_50':
            network = self.resnet_v2_50()
        elif self.model_name == 'resnet_v2_101':
            network = self.resnet_v2_101()
        elif self.model_name == 'resnet_v2_152':
            network = self.resnet_v2_152()
        elif self.model_name == 'densenet_121':
            network = densenet_121()
        elif self.model_name == 'densenet_169':
            network = self.densenet_169()
        elif self.model_name == 'densenet_201':
            network = self.densenet_201()
        elif self.model_name == 'inception_v3':
            network = self.inception_v3()
        elif self.model_name == 'inception_v4':
            network = self.inception_v4()
        elif self.model_name == 'efficientnet':
            network = self.efficientnet()

        model = Sequential()
        model.add(network)
        model.add(Dense(2048, activation='relu'))
        model.add(Dense(self.num_class, activation='softmax'))
        model.summary()

        return model

class Fine_tunning:
    def __init__(self, train_path, model_name, epoch, multi_gpu=0):
        self.data = Import_data(train_path) #데이터 전처리 및 셋팅할 객체 가져옴
        self.train_data, self.val_data = self.data.train() # 데이터 전처리 및 셋팅 진행
        self.load_model = Load_model(train_path, model_name) #모델 만들 객체 가져옴
        self.multi_gpu = multi_gpu
        self.epoch = epoch
        self.model_name = model_name
        self.train_path = train_path

    def training(self):
        print("training")
        data_name = self.train_path.split('/') #경로를 /로 나눔
        data_name = data_name[len(data_name)-2] #데이터의 이름을 가져옴 --> but 향후 경로에 맞게 고칠필요 있을까
        print("data_name : "+data_name)
        optimizer = keras.optimizers.SGD(learning_rate=0.001, decay=1e-5, momentum=0.999, nesterov=True)
        #(실제 값을 반환하는 학습률, ?, 관련 방향으로 경사 하강을 가속하고 진동을 감쇠시킴, 네스테로프 운동량을 적용할지 여부)
        model = self.load_model.build_network() #레이어 적용할 메서드 가져옴
        save_folder = './model_saved/' + data_name + '/' + self.model_name + '_' + str(self.epoch) + '/' #모델 저장 경로
        if not os.path.exists(save_folder):
            os.makedirs(save_folder) #경로가 존재하지 않다면 알아서 생성됨
        check_point = ModelCheckpoint(save_folder + 'model-{epoch:03d}-{acc:03f}-{val_acc:03f}.h5', verbose=1,
                                      monitor='val_acc', save_best_only=True, mode='auto')#모델 저장할 때 사용하는 콜백 함수
        # (모델 저장경로, 저장되었다고 화면에 표시됨, val_acc가 가장 클때 저장하고 싶음, 모니터 되고 있는 값 중 가장 좋은 값이 저장됨, 모델이 알아서 min과 max를 판단하여 저장
        if self.multi_gpu == 0:
            model.compile(loss='categorical_crossentropy',
                          optimizer=optimizer,
                          metrics=['acc']) #(로스 함수를 지정하기 위한 용도, 신경망 최적화를 위한 알고리즘을 지정하는 용도, AI의 성능을 채점하기 위한 기준을 지정하는 용도)
            history = model.fit_generator(
                self.train_data,
                steps_per_epoch=self.train_data.samples / self.train_data.batch_size, #한 에폭에 사용한 스텝 수(한 에폭을 배치를 몇번을 가져와 훈련할 것인지)
                epochs=self.epoch, #에폭수
                validation_data=self.val_data, #검증 데이터셋을 제공할 제네레이터 지정
                validation_steps=self.val_data.samples / self.val_data.batch_size, #한 에폭 종료시마다 검증할 때 사용되는 검증 스텝 수
                callbacks=[check_point], #콜백 함수를 체크 포인트로 받음
                verbose=1#완료되었을 때 완료됐다고 뜸 #fit()함수 이용하여 인공신경망 학습 개시
            )
        else:
            with tf.device('/cpu:0'): #with절 내에서 자원 사용 - 특정작업을 원하는 디바이스에 배치(시스템의 cpu를 지정함)
                cpu_model = model
            model = multi_gpu_model(cpu_model, gpus=self.multi_gpu) #gpu를 사용하기 위한 메서드(, 사용하고자 하는 gpu개수)
            model.summary()
            model.compile(loss='categorical_crossentropy',
                          optimizer=optimizer,
                          metrics=['acc'])
            history = model.fit_generator(
                self.train_data,
                steps_per_epoch=self.train_data.samples / self.train_data.batch_size,
                epochs=self.epoch,
                validation_data=self.val_data,
                validation_steps=self.val_data.samples / self.val_data.batch_size,
                callbacks=[check_point],
                verbose=1)
        print("training end")
        return history

    def save_accuracy(self, history):
        print("save_accuracy")
        data_name = self.train_path.split('/')
        data_name = data_name[len(data_name)-2]
        save_folder = './model_saved/' + data_name + '/' + self.model_name + '_' + str(self.epoch) + '/'
        acc = history.history['acc'] #기록에서 정확도를 추출해 옴
        val_acc = history.history['val_acc']
        loss = history.history['loss']
        val_loss = history.history['val_loss']
        epochs = range(len(acc)) #acc의 길이만큼 리스트 형태로 epochs에 넣음
        epoch_list = list(epochs) #epochs를 진짜 리스트로 형변환

        #결과를 csv파일로 저장
        df = pd.DataFrame({'epoch': epoch_list, 'train_accuracy': acc, 'validation_accuracy': val_acc},
                          columns=['epoch', 'train_accuracy', 'validation_accuracy'])
        df_save_path = save_folder + 'accuracy.csv'
        df.to_csv(df_save_path, index=False, encoding='euc-kr')

        plt.plot(epochs, acc, 'b', label='Training acc')
        plt.plot(epochs, val_acc, 'r', label='Validation acc')
        plt.title('Training and validation accuracy')
        plt.legend()
        save_path = save_folder + 'accuracy.png'
        plt.savefig(save_path)
        plt.cla()

        plt.plot(epochs, loss, 'b', label='Training loss')
        plt.plot(epochs, val_loss, 'r', label='Validation loss')
        plt.title('Training and validation loss')
        plt.legend()
        save_path = save_folder + 'loss.png'
        plt.savefig(save_path)
        plt.cla()

        name_list = os.listdir(save_folder)
        h5_list = []
        for name in name_list:
            if '.h5' in name:
                h5_list.append(name)
        h5_list.sort()
        h5_list = [save_folder + name for name in h5_list]
        for path in h5_list[:len(h5_list) - 1]:
            os.remove(path)
        K.clear_session()
