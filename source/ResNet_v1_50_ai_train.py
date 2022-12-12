#project
from ops_1.ResNet_ops_1 import *

train_path = '../data/label/cat/Sequestrum/train/' #경로 마지막에 반드시 '/'를 기입해야합니다.
model_name = 'resnet_v1_50'
epoch = 3

if __name__ == '__main__':
    print("start")
    # 현재 파일 이름
    print(__file__)

    # 현재 파일 실제 경로
    print(os.path.realpath(__file__))

    # 현재 파일 절대 경로
    print(os.path.abspath(__file__))

    # 현재 폴더 경로; 작업 폴더 기준
    print(os.getcwd())

    # 현재 파일의 폴더 경로; 작업 파일 기준
    print(os.path.dirname(os.path.realpath(__file__)))

    print(os.listdir(os.getcwd()))

    print("file : " +os.path.abspath("../data/label"))
    print("what")

    fine_tunning = Fine_tunning(train_path=train_path,
                                model_name=model_name,
                                epoch=epoch) #훈련할 데이터의 경로, 사용할 신경구조, 에폭수를 지정
    history = fine_tunning.training() #데이터 훈련 시작
    print("training end")
    fine_tunning.save_accuracy(history) #기록 저장