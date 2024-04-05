import torch
import torch.nn as nn
from torch.utils.data import Dataset
from torchvision import datasets
from torchvision.transforms import ToTensor
from torch.utils.data import DataLoader
import torchaudio

from urbansounddataset import UrbanSoundDataset # urbansounddataset.py 파일내에 있는 UrbanSoundDataset의 클래스를 가져와서 사용한다. 
from cnn import CNNNetwork # cnn.py에서 CNNNetwork class import  


BATCH_SIZE = 128
EPOCHS = 10
LEARNING_RATE = .001
ANNOTATIONS_FILE =  "UrbanSound8K/UrbanSound8K/metadata/UrbanSound8K.csv"
AUDIO_DIR = "UrbanSound8K/UrbanSound8K/audio"


SAMPLE_RATE=  22050 # 가지고 싶은 샘플 수를 결정 16000 -> 22050로 변경
NUM_SAMPLES = 22050



def create_data_loader(train_data,batch_size):
    train_dataloader = DataLoader(train_data,batch_size=batch_size)
    return train_dataloader


def train_one_epoch(model,data_loader,loss_fn,optimiser,device):
    # 데이터의 모든 샘플을 반복하는 루프를 만들고 싶다.
    # 각 반복에서 새로운 샘플 배치를 얻는다.
    for inputs, targets in data_loader:
        inputs,targets = inputs.to(device), targets.to(device)

        # calculate loss
        preditions = model(inputs)
        loss = loss_fn(preditions,targets)

        # back_propagate loss and update weights
        optimiser.zero_grad() # 매 반복마다 업데이트
        loss.backward() # 역전파
        optimiser.step() 

    print(f"Loss : {loss.item()}") # 마지막 배치에 대한 손실을 인쇄
        


def train(model,data_loader,loss_fn,optimiser,device,epochs):
    for i in range(epochs):
        print(f"Epoch {i+1}")

        train_one_epoch(model,data_loader,loss_fn,optimiser,device)
        print("---------------------")

    print("training is done")


if __name__ == "__main__":
    if torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"



    # 데이터에 적용할 mel_spectrogram를 인스턴스화 해야한다. 그리고 이는 UrbanSoundDataset에 전달됨 
    mel_spectrogram = torchaudio.transforms.MelSpectrogram(
          sample_rate=SAMPLE_RATE,
          n_fft=1024,
          hop_length=512,
          n_mels=64
    )

    # ms = mel_spectrogram(siganl)

    usd = UrbanSoundDataset(ANNOTATIONS_FILE,
                            AUDIO_DIR,
                            mel_spectrogram, # 위에서 정의한 mel_spectrogram을 사용
                            SAMPLE_RATE,
                            NUM_SAMPLES,
                            device)



    # create data loader : 상당히 무거운 데이터 세트도 로드 가능
    train_dataloader = create_data_loader(usd,BATCH_SIZE)

    # 
    cnn = CNNNetwork().to(device)
    print(cnn)

    loss_fn= nn.CrossEntropyLoss()
    optimiser = torch.optim.Adam(cnn.parameters(),
                                 lr = LEARNING_RATE)

    # 4. train model
    train(cnn,train_dataloader,loss_fn,optimiser,device,EPOCHS)
    
    # 5. save trained model 
    torch.save(cnn.state_dict(),"cnn.pth")
    # state_dict : 훈련된 레이어 및 매개변수에 관한 정보를 포함하는 딕셔너리
    print("Model trained and stored at cnn.pth") 