import torch
import torch.backends.cudnn
from torch import nn
from torch.nn import functional as F


class BaseCNN(nn.Module):
    '''
    CNN Replication of the model from the paper (Base Model)

    WORKS WITH RAW AUDIO

    Based on paper: https://ieeexplore.ieee.org/document/6854950
    End-to-end learning for music audio [Submitted on 14 May 2014]
    By Sander Dieleman and Benjamin Schrauwen
    '''

    def __init__(self, length: int, stride = int):
        super().__init__()

        #strided convolution
        self.conv0 = nn.Conv1d(in_channels = 1, out_channels=32, kernel_size=length, stride=stride) 
        self.initialise_layer(self.conv0)

        #replicated network architecture from the paper

        #first convolution + batch norm
        self.conv1 = nn.Conv1d(in_channels=32, out_channels=32, kernel_size=8, padding = 'same')
        self.initialise_layer(self.conv1)
        self.bn1 = nn.BatchNorm1d(num_features = 32, eps = 1e-5, momentum = 0.1, affine = True, track_running_stats = True)

        #first pooling
        self.pool1 = nn.MaxPool1d(kernel_size=4)

        #second convolution + batch norm
        self.conv2 = nn.Conv1d(in_channels=32, out_channels=32, kernel_size=8, padding = 'same')
        self.initialise_layer(self.conv2)
        self.bn2 = nn.BatchNorm1d(num_features = 32, eps = 1e-5, momentum = 0.1, affine = True, track_running_stats = True)

        #second pooling
        self.pool2 = nn.MaxPool1d(kernel_size=4)

        #fully connected layers
        self.fc1 = nn.Linear(32, 100)
        self.initialise_layer(self.fc1)

        self.bn3 = nn.BatchNorm1d(num_features = 100, eps = 1e-5, momentum = 0.1, affine = True, track_running_stats = True)

        self.fc2 = nn.Linear(100, 50)
        self.initialise_layer(self.fc2)

        self.dropout1 = nn.Dropout(p=0.5)

    def forward(self, input: torch.Tensor) -> torch.Tensor:

        batch_size = input.size(0)

        #flatten the tensor to combine - batch size and clips
        x = torch.flatten(input, start_dim=0, end_dim=1)

        x = F.relu(self.conv0(x))

        x = F.relu(self.conv1(x))
        x = self.bn1(x)
        x = self.pool1(x)
        x = F.relu(self.conv2(x))
        x = self.bn2(x)
        x = self.pool2(x)

        #dynamic calculation to reshape input for fc1
        pool2_B , pool2_C, pool2_F = x.shape
        pool_avg = nn.AvgPool1d(pool2_F)
        x = pool_avg(x)
        x = x.reshape(pool2_B, pool2_C)

        x = torch.flatten(x, start_dim=1)

        x = F.relu(self.fc1(x))
        x = self.bn3(x)
        x = self.dropout1(x)
        x = torch.sigmoid(self.fc2(x))
        
        x = torch.reshape(x, (batch_size, -1, 50))
        x = x.mean(dim=1)

        return x
    
    @staticmethod
    def initialise_layer(layer):
        if hasattr(layer, "bias"):
            nn.init.zeros_(layer.bias)
        if hasattr(layer, "weight"):
            nn.init.kaiming_normal_(layer.weight)

class BaseCNNNonInitialised(nn.Module):
    '''
    CNN Replication of the model from the paper (Base Model) WITHOUT KAIMING INITIALISATION

    WORKS WITH RAW AUDIO

    Based on paper: https://ieeexplore.ieee.org/document/6854950
    End-to-end learning for music audio [Submitted on 14 May 2014]
    By Sander Dieleman and Benjamin Schrauwen
    '''

    def __init__(self, length: int, stride = int):
        super().__init__()

        #strided convolution
        self.conv0 = nn.Conv1d(in_channels = 1, out_channels=32, kernel_size=length, stride=stride) 
        

        #replicated network architecture from the paper

        #first convolution + batch norm
        self.conv1 = nn.Conv1d(in_channels=32, out_channels=32, kernel_size=8, padding = 'same')
        
        self.bn1 = nn.BatchNorm1d(num_features = 32, eps = 1e-5, momentum = 0.1, affine = True, track_running_stats = True)

        #first pooling
        self.pool1 = nn.MaxPool1d(kernel_size=4)

        #second convolution + batch norm
        self.conv2 = nn.Conv1d(in_channels=32, out_channels=32, kernel_size=8, padding = 'same')
        
        self.bn2 = nn.BatchNorm1d(num_features = 32, eps = 1e-5, momentum = 0.1, affine = True, track_running_stats = True)

        #second pooling
        self.pool2 = nn.MaxPool1d(kernel_size=4)

        #fully connected layers
        self.fc1 = nn.Linear(32, 100)
        

        self.bn3 = nn.BatchNorm1d(num_features = 100, eps = 1e-5, momentum = 0.1, affine = True, track_running_stats = True)

        self.fc2 = nn.Linear(100, 50)
       

        self.dropout1 = nn.Dropout(p=0.5)

    def forward(self, input: torch.Tensor) -> torch.Tensor:

        batch_size = input.size(0)

        #flatten the tensor to combine - batch size and clips
        x = torch.flatten(input, start_dim=0, end_dim=1)

        x = F.relu(self.conv0(x))

        x = F.relu(self.conv1(x))
        x = self.bn1(x)
        x = self.pool1(x)
        x = F.relu(self.conv2(x))
        x = self.bn2(x)
        x = self.pool2(x)

        #dynamic calculation to reshape input for fc1
        pool2_B , pool2_C, pool2_F = x.shape
        pool_avg = nn.AvgPool1d(pool2_F)
        x = pool_avg(x)
        x = x.reshape(pool2_B, pool2_C)

        x = torch.flatten(x, start_dim=1)

        x = F.relu(self.fc1(x))
        x = self.bn3(x)
        x = self.dropout1(x)
        x = torch.sigmoid(self.fc2(x))
        
        x = torch.reshape(x, (batch_size, -1, 50))
        x = x.mean(dim=1)

        return x
    
    @staticmethod
    def initialise_layer(layer):
        if hasattr(layer, "bias"):
            nn.init.zeros_(layer.bias)
        if hasattr(layer, "weight"):
            nn.init.kaiming_normal_(layer.weight)

class ChunkResCNN(nn.Module):
    '''
    Extension of CNN to process chunks of audio with residual connections

    WORKS WITH SPECTROGRAMS

    Based on paper: https://arxiv.org/abs/2006.00751v1
    Evaluation of CNN-based Automatic Music Tagging Models [Submitted on 1 Jun 2020]
    By Minz Won, Andres Ferraro, Dmitry Bogdanov, Xavier Serra
    '''
    def __init__(self,
                n_channels=128,
                n_class=50):
        super(ChunkResCNN, self).__init__()

        #cnn
        self.layer1 = ResCNN2d(1, n_channels, stride=2)
        self.layer2 = ResCNN2d(n_channels, n_channels, stride=2)
        self.layer3 = ResCNN2d(n_channels, n_channels*2, stride=2)
        self.layer4 = ResCNN2d(n_channels*2, n_channels*2, stride=2)
        self.layer5 = ResCNN2d(n_channels*2, n_channels*2, stride=2)
        self.layer6 = ResCNN2d(n_channels*2, n_channels*2, stride=2)
        self.layer7 = ResCNN2d(n_channels*2, n_channels*4, stride=2)

        #fc
        self.fc1 = nn.Linear(n_channels*4, n_channels*4)
        self.bn = nn.BatchNorm1d(n_channels*4)
        self.fc2 = nn.Linear(n_channels*4, n_class)
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.layer5(x)
        x = self.layer6(x)
        x = self.layer7(x)
        x = x.squeeze(2)

        #global max pooling
        if x.size(-1) != 1:
            x = nn.MaxPool1d(x.size(-1))(x)
        x = x.squeeze(2)

        x = F.relu(self.fc1(x))
        x = self.bn(x)
        x = self.dropout(x)
        x = torch.sigmoid(self.fc2(x))

        return x
    
    @staticmethod
    def initialise_layer(layer):
        if hasattr(layer, "bias"):
            nn.init.zeros_(layer.bias)
        if hasattr(layer, "weight"):
            nn.init.kaiming_normal_(layer.weight)

class CRNN(nn.Module):
    '''
    Extension of CRNN to process audio

    WORKS WITH SPECTROGRAMS

    Based on paper: https://arxiv.org/abs/2006.00751v1
    Evaluation of CNN-based Automatic Music Tagging Models [Submitted on 1 Jun 2020]
    By Minz Won, Andres Ferraro, Dmitry Bogdanov, Xavier Serra
    '''
    def __init__(self,
                n_channels=128,
                n_class=50):
        super(CRNN, self).__init__()

        self.layer1 = Conv2d(1, n_channels, stride=2)
        self.layer2 = Conv2d(n_channels, n_channels, stride=2)
        self.layer3 = Conv2d(n_channels, n_channels * 2, stride=2)
        self.layer4 = Conv2d(n_channels * 2, n_channels * 2, stride=2)
        self.layer5 = nn.GRU(n_channels * 2, n_channels, 2, batch_first=True)

        self.fc1 = nn.Linear(11, 100)
        self.bn = nn.BatchNorm1d(100)
        self.fc2 = nn.Linear(100, n_class)
        self.dropout = nn.Dropout(0.5)
        self.relu = nn.ReLU()

    def forward(self, x):

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)

        x = x.squeeze(2)
        x = x.permute(0, 2, 1)
        x, _ = self.layer5(x)

        x = nn.AdaptiveMaxPool1d(1)(x)
        x = x.view(x.size(0), -1)

        x = F.relu(self.fc1(x))
        x = self.bn(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = torch.sigmoid(x)
        return x

#helper classes
class ResCNN2d(nn.Module):
    def __init__(self, input_channels, output_channels, kernel_size=3, stride=1, padding=1):
        super(ResCNN2d, self).__init__()
        self.conv1 = nn.Conv2d(input_channels, output_channels, kernel_size, stride, padding)
        self.bn1 = nn.BatchNorm2d(output_channels)
        self.conv2 = nn.Conv2d(output_channels, output_channels, kernel_size, padding=padding)
        self.bn2 = nn.BatchNorm2d(output_channels)

        self.shortcut = nn.Sequential()
        if stride != 1 or input_channels != output_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(input_channels, output_channels, kernel_size=1, stride=stride),
                nn.BatchNorm2d(output_channels)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out

class Conv2d(nn.Module):
    def __init__(self, input_channels, output_channels, shape=3, stride=1, pooling=2, pooling_stride=2):
        super(Conv2d, self).__init__()
        self.conv = nn.Conv2d(input_channels, output_channels, shape, stride=stride, padding=shape//2)
        self.bn = nn.BatchNorm2d(output_channels)
        self.relu = nn.ReLU()
        self.mp = nn.MaxPool2d(pooling, stride=pooling_stride)

    def forward(self, x):
        out = self.mp(self.relu(self.bn(self.conv(x))))
        return out