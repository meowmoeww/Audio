import torch
import torch.nn as nn
from torch.utils.data import Dataset
from torchvision import datasets
from torchvision.transforms import ToTensor
from torch.utils.data import DataLoader


BATCH_SIZE = 128
EPOCHS = 10
LEARNING_RATE = .001

#3. build model
class FeedForwardNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.flatten = nn.Flatten()
        # Sequential: 순차를 고려하여 여러개의 layer 쌓기 가능
        self.dense_layers= nn.Sequential(
            # (28,28) - flatten -> (28*28,)
            nn.Linear(28*28,256),
            nn.ReLU(), #activation
            nn.Linear(256,10) # 10개의 클래스이기 때문에
        )
        self.softmax = nn.Softmax(dim=1)

    def forward(self,input_data):
        flattened_data = self.flatten(input_data)
        logits = self.dense_layers(flattened_data)
        predictions = self.softmax(logits)
        return predictions



# 1. dowload dataset
def dowload_datasets():
    train_data = datasets.MNIST(
        root="data",
        download=True,
        train=True,
        transform=ToTensor()
    )

    validation_data = datasets.MNIST(
        root="data",
        download=True,
        train=False,
        transform=ToTensor()
    )

    return train_data,validation_data

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
    # dowload dataset
    train_data ,_ = dowload_datasets()
    print("MNIST dataset download")

    # 2. create data loader : 상당히 무거운 데이터 세트도 로드 가능
    train_dataloader = create_data_loader(dataset=train_data, batch_size=BATCH_SIZE)

    # 3. build model
    if torch.cuda.is_available():
        device ='cuda'
    else:
        device = 'cpu'
    
    print(f"Using {device} device")
    feed_forward_net = FeedForwardNet().to(device)

    loss_fn= nn.CrossEntropyLoss()
    optimiser = torch.optim.Adam(feed_forward_net.parameters(),
                                 lr = LEARNING_RATE)

    # 4. train model
    train(feed_forward_net,train_dataloader,loss_fn,optimiser,device,EPOCHS)
    
    # 5. save trained model 
    torch.save(feed_forward_net.state_dict(),"feedforwardnet.pth")
    # state_dict : 훈련된 레이어 및 매개변수에 관한 정보를 포함하는 딕셔너리
    print("Model trained and stored at feedforwardnet.pth") 