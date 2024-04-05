# 훈련한 모델을 다시 불러온 다음 inference 하기 
import torch
# train.py에서 정의한 FeedForwardNet과 download_mnist_datasets가져오겠다 
from train import FeedForwardNet, dowload_datasets

class_mapping = [
    "0",
    "1",
    "2",
    "3",
    "4",
    "5",
    "6",
    "7",
    "8",
    "9",
]   
def predict(model,input,target,class_mapping):
    # inference를 할 때는 무조건 eval로 돌리기 
    model.eval()
    with torch.no_grad():
        predictions = model(input)
        # predictions : Tensor (2 dim)->(N,output_class) 여기서는 (1,10) 
        # 가장 높은 값을 갖는 인덱스에 관심이 있다.
        predicted_index = predictions[0].argmax(0)
        predicted = class_mapping[predicted_index]
        expected = class_mapping[target]

    return predicted, expected






if __name__ == "__main__":
    # load back the model
    feed_forward_net = FeedForwardNet()
    state_dict = torch.load("feedforwardnet.pth")
    # 모델의 정보를 저장해둔 state_dict의 정보를 다시 net에게 전달
    feed_forward_net.load_state_dict(state_dict)

    # load MNIST validation dataset
    _, validation_data = dowload_datasets()

    # get a sample from the validation dataset for inference
    input, target = validation_data[0][0],validation_data[0][1]

    # make an inference
    predicted,expected = predict(feed_forward_net,input,target,class_mapping)
    # class_mapping: 신경망은 우리각 다루고 있는 클래스의 미름에 대해 아무 것도 모르기 때문에 정수만 사용하고
    # 간단한 목록으로 만든다 

    print(f"Predicted : {predicted}, expected : {expected}")

