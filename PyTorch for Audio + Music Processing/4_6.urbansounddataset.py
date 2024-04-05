from torch.utils.data import Dataset
import pandas as pd 
import torchaudio 
import os 
import torch 

class UrbanSoundDataset(Dataset):
    def __init__(self,
                 annotations_file, 
                 audio_dir,
                 transformation,
                 target_sample_rate,
                 num_samples,
                 device):
            # audio_dir : Urban sound 8k의 디렉토리에 대한 경로
            self.annotations = pd.read_csv(annotations_file)
            self.audio_dir = audio_dir
            self.device = device
            self.transformation = transformation.to(self.device)
            self.target_sample_rate = target_sample_rate
            self.num_samples = num_samples
            

    def __len__(self):
         return len(self.annotations)

    def __getitem__(self,index):    
        # a_list[1] -> a_list.__getitem__(1)
        audio_sample_path = self._get_audio_sample_path(index) # 오디오 샘플의 경로
        label = self._get_audio_sample_label(index)
        signal,sr = torchaudio.load(audio_sample_path) # 파형을 mel spectrogram으로 만들고 싶은것
        signal = signal.to(self.device)
        # signal -> (num_channels,samples) -> (2,10000) -> (1,16000)
        signal = self._resample_if_necessary(signal,sr) # 필요하면 resamling 진행
        signal = self._mix_down_if_necessary(signal) # 신호가 여러개면 모노로 혼합 
        # 신호가 예상한 것보다 훨씬 길다면 신호를 자르고 초기 샘플을 예상 샘플 수만큼만 남겨 두는 것
        signal = self._cut_if_necessary(signal)
        # 우리가 기대하는 샘플보다 더 적은 샘플을 가지는 신호에 대해서는 남는 부분에 대해 zero padding을 진행해준다.
        signal = self._right_pad_if_necessary(signal)
        signal = self.transformation(signal) # 멜 스펙트로그램으로 변환
        return signal, label 
    

    def _cut_if_necessary(self,signal):
         # signal -> Tensor => (1, ) # (number of channel , num_samples)
        if signal.shape[1] > self.num_samples:
            signal = signal[:,:self.num_samples]
        return signal 
    
    def _right_pad_if_necessary(self,signal):
        length_signal = signal.shape[1]
        if length_signal < self.num_samples: # 신호에 있는 샘플 수가 예상되는 샘플 수보다 적을 때 
            # [1,1,1] -> [1,1,1,0,0]
            num_missing_samples = self.num_samples -  length_signal # 해당 신호가 예상하는 샘플 수가 있기 위해서 필요한 샘플의 수 
            last_dim_padding = (0,num_missing_samples)
            signal = torch.nn.functional.pad(signal,last_dim_padding) # 필요한 만큼 padding 해주기 
        return signal  
              

    def _resample_if_necessary(self,signal,sr):
        # 모든 신호마다 샘플링 속도가 같은게 아니므로, 모두 같은 속도를 가질 수 있게 재샘플링을 진행해야 한다. 
        if sr != self.target_sample_rate:
            resampler = torchaudio.transforms.Resample(sr,self.target_sample_rate)
            signal = resampler(signal)
        return signal
    
    def _mix_down_if_necessary(self, signal):
         # 만약 신호의 채널이 1개가 아니라면 신호를 믹스다운 해준다.
        if signal.shape[0] > 1: # (2,1000) 이럴 경우 
            signal = torch.mean(signal,dim=0,keepdim=True)
        return signal 
    

    def _get_audio_sample_path(self,index):
         # annotations : 데이터 프레임 
         fold = f"fold{self.annotations.iloc[index,5]}" # 5는 fold 열을 의미
         path = os.path.join(self.audio_dir,fold,self.annotations.iloc[index,0])
         return path 
    
    def _get_audio_sample_label(self,index):
         return self.annotations.iloc[index,6]
    

if __name__ == "__main__":
    ANNOTATIONS_FILE =  "UrbanSound8K/UrbanSound8K/metadata/UrbanSound8K.csv"
    AUDIO_DIR = "UrbanSound8K/UrbanSound8K/audio"


    SAMPLE_RATE=  22050 # 가지고 싶은 샘플 수를 결정 16000 -> 22050로 변경
    NUM_SAMPLES = 22050

    if torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"

    print(f"Using device {device}")

    mel_spectrogram = torchaudio.transforms.MelSpectrogram(
          sample_rate=SAMPLE_RATE,
          n_fft=1024,
          hop_length=512,
          n_mels=64
    )

     # ms = mel_spectrogram(siganl)

    usd = UrbanSoundDataset(ANNOTATIONS_FILE,
                            AUDIO_DIR,
                            mel_spectrogram,
                            SAMPLE_RATE,
                            NUM_SAMPLES,
                            device)

    print(f"There are {len(usd)} samples in the dataset")

    signal,label =usd[0] #usd의 0번째 데이터를 가지고 오는 것

      
         