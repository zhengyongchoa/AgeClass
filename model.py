import torch ,math
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
from torch.autograd import Variable
import options as opt


class Simple_Classify(nn.Module): # inputDim =[1, 21 , 63]
    def __init__(self,  nClasses= opt.nClasses):
        super(Simple_Classify, self).__init__()
        self.nClasses = nClasses
        # self.conv1 = nn.Conv1d(in_channels=21,out_channels=1024,kernel_size=3)
        # self.conv2 = nn.Conv1d(in_channels=1024,out_channels=512,kernel_size=3)
        # self.conv3 = nn.Conv1d(in_channels=512,out_channels=256,kernel_size=3)

        self.conv1 = nn.Sequential(
            nn.Conv1d(in_channels=21, out_channels= 1024, kernel_size=3),
            nn.BatchNorm1d(1024, momentum=0.01),
            nn.ReLU(inplace=True),
            nn.Conv1d(in_channels=1024 ,out_channels= 512, kernel_size=3),
            nn.MaxPool1d(kernel_size=2),
        )
        self.conv2 = nn.Sequential(
            nn.Conv1d(in_channels= 512, out_channels= 256, kernel_size= 3 ),
            nn.BatchNorm1d(256, momentum=0.01),
            nn.ReLU(inplace=True),
            nn.Conv1d(in_channels=256, out_channels=128 ,kernel_size=3),
            nn.MaxPool1d(kernel_size=2),
        )
        #卷积的输出尺寸
        self.lstm = nn.LSTM(input_size= 12, hidden_size=128,num_layers= 2, batch_first=True, bidirectional=True)
        # self.lstm = nn.LSTM(input_size= 12, hidden_size=128)

        self.fc6 = nn.Sequential(nn.Linear(128* 256, 128* 8), nn.BatchNorm1d(128*8), nn.ReLU(True))
        # self.fc6 = nn.Sequential(nn.Linear(128* 128, 128* 8), nn.BatchNorm1d(128*8), nn.ReLU(True))

        self.fc7 = nn.Sequential(nn.Linear(128*8,64), nn.BatchNorm1d(64), nn.ReLU(True))
        self.fc8 = nn.Sequential(nn.Linear(64, self.nClasses), nn.ReLU(True))

    def forward(self, x):
        in_size = x.size(0)
        x = self.conv1(x)
        # print(x.shape)
        x = self.conv2(x)
        # print(x.size())
        x ,_ = self.lstm(x)
        # print(x.size())
        x = x.reshape(in_size , -1)

        # print(x.size())
        x = self.fc6(x)
        # print(x.size())
        x = self.fc7(x)
        x = self.fc8(x)
        # print(x.size())
        # print(x[0])
        return x

    def weights_init(self , m):

        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.kernel_size[2] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()

            elif isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()

            elif isinstance(m, nn.Conv1d):
                n = m.kernel_size[0] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()

            elif isinstance(m, nn.BatchNorm3d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

            elif isinstance(m, nn.BatchNorm1d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()


class Age_classify(nn.Module): # inputDim =[1, 21 , 63]
    def __init__(self, nClasses=3):
        super(Age_classify , self).__init__()
        self.nClasses = nClasses
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=24, kernel_size=5)
        self.conv2 = nn.Conv2d(in_channels=24, out_channels= 64, kernel_size=5)
        self.bilstm = nn.Conv1d(in_channels=512, out_channels=256, kernel_size=2)
        self.conv3 = nn.Conv1d(in_channels=256, out_channels=128, kernel_size=2)

        # 卷积的输出尺寸
        self.fc6 = nn.Sequential(nn.Linear(64 * 15, 128 * 2), nn.BatchNorm1d(128 * 2), nn.ReLU(True))
        self.fc7 = nn.Sequential(nn.Linear(128 * 2, 128), nn.BatchNorm1d(128), nn.ReLU(True))
        self.fc8 = nn.Sequential(nn.Linear(128, self.nClasses))
        self.mp = nn.MaxPool2d(2)
        self.fc = nn.Linear(320, 10)

    def forward(self, x):
        # print(x.shape)
        x = self.conv1(x)
        # print(x.shape)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)
        # print(x.size())
        x = x.view(opt.batch_size, -1)
        x = self.fc6(x)
        x = self.fc7(x)
        # print(x.size())
        x = self.fc8(x)

        return x



