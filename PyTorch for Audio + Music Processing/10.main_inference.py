# 훈련한 모델을 다시 불러온 다음 inference 하기 
import torch
# train.py에서 정의한 FeedForwardNet과 download_mnist_datasets가져오겠다 
from cnn import CNNNetwork
from urbansounddataset import UrbanSoundDataset
from main_train import AUDIO_DIR,ANNOTATIONS_FILE,SAMPLE_RATE,NUM_SAMPLES
import torchaudio

class_mapping = [
    "air_conditioner",
    "car_horn",
    "children_playing",
    "dog_bark",
    "drilling",
    "engine_idling",
    "gun_shot",
    "jackhammer",
    "siren",
    "street_music",
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
    cnn = CNNNetwork()
    state_dict = torch.load("cnn.pth")
    # 모델의 정보를 저장해둔 state_dict의 정보를 다시 net에게 전달
    cnn.load_state_dict(state_dict)

    # load urbansound dataset
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
                            "cpu")

    # get a sample from the urban sound dataset for inference
    input, target = usd[0][0],usd[0][1] # 3차원을 가지는 텐서플로 [num_channels,fr,time]
    #unsqueze_  기존 텐서의 데이터는 그대로 유지하면서 새로운 차원(크기가 1)만 추가됩니다.
    input = input.unsqueeze(0) # [num_channels,fr,time] -> [1,num_channels,fr,time] 0번째 인덱스에 차원을 추가하겠다.

    
    # make an inference
    predicted,expected = predict(cnn,input,target,class_mapping)
    # class_mapping: 신경망은 우리각 다루고 있는 클래스의 미름에 대해 아무 것도 모르기 때문에 정수만 사용하고
    # 간단한 목록으로 만든다 

    print(f"Predicted : {predicted}, expected : {expected}")

