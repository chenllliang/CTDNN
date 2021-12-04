import torch.nn as nn
import torch.nn.functional as F
import torch
from FTDNN import FTDNN


class TDNN(nn.Module):

    def __init__(
            self,
            input_dim=13,
            output_dim=512,
            context_size=5,
            stride=1,
            dilation=1,
            batch_norm=True,
            dropout_p=0.0
    ):
        '''
        TDNN as defined by https://www.danielpovey.com/files/2015_interspeech_multisplice.pdf
        Affine transformation not applied globally to all frames but smaller windows with local context
        batch_norm: True to include batch normalisation after the non linearity

        Context size and dilation determine the frames selected
        (although context size is not really defined in the traditional sense)
        For example:
            context size 5 and dilation 1 is equivalent to [-2,-1,0,1,2]
            context size 3 and dilation 2 is equivalent to [-2, 0, 2]
            context size 1 and dilation 1 is equivalent to [0]
        '''
        super(TDNN, self).__init__()
        self.context_size = context_size
        self.stride = stride
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.dilation = dilation
        self.dropout_p = dropout_p
        self.batch_norm = batch_norm

        self.kernel = nn.Linear(input_dim * context_size, output_dim)
        self.nonlinearity = nn.ReLU()
        if self.batch_norm:
            self.bn = nn.BatchNorm1d(output_dim)
        if self.dropout_p:
            self.drop = nn.Dropout(p=self.dropout_p)

    def forward(self, x):
        '''
        input: size (batch, seq_len, input_features)
        outpu: size (batch, new_seq_len, output_features)
        '''

        _, _, d = x.shape
        assert (d == self.input_dim), 'Input dimension was wrong. Expected ({}), got ({})'.format(self.input_dim, d)
        x = x.unsqueeze(1)

        # Unfold input into smaller temporal contexts
        x = F.unfold(
            x,
            (self.context_size, self.input_dim),
            stride=(1, self.input_dim),
            dilation=(self.dilation, 1)
        )

        # N, output_dim*context_size, new_t = x.shape
        x = x.transpose(1, 2)
        x = self.kernel(x)
        x = self.nonlinearity(x)

        if self.dropout_p:
            x = self.drop(x)

        if self.batch_norm:
            x = x.transpose(1, 2)
            x = self.bn(x)
            x = x.transpose(1, 2)

        return x



class TDXN(nn.Module):
    def __init__(self,in_dim):
        super(TDXN, self).__init__()
        self.frame1 = TDNN(input_dim=in_dim, output_dim=512, context_size=5, dilation=1)
        self.frame2 = TDNN(input_dim=512, output_dim=512, context_size=3, dilation=2)
        self.frame3 = TDNN(input_dim=512, output_dim=512, context_size=3, dilation=3)
        self.frame4 = TDNN(input_dim=512, output_dim=512, context_size=1, dilation=1)
        self.frame5 = TDNN(input_dim=512, output_dim=1500, context_size=1, dilation=1)
        self.fc1 = nn.Linear(2078, 1024)
        self.fc2 = nn.Linear(1024, 512)
        self.fc3 = nn.Linear(512, 10)

        self.bn1 = nn.BatchNorm1d(512)
        self.bn2 = nn.BatchNorm1d(512)

    def forward(self,x):

        o1 = self.frame1(x)
        o2 = self.frame2(o1)
        o3 = self.frame3(o2)
        o4 = self.frame4(o3)
        o5 = self.frame5(o4)
        o6 = self.statistic_pooling(o5)
        # o7 = F.relu(self.bn1(self.fc1(o6)))
        # o8 = F.relu(self.bn2(self.fc2(o7)))

        o7 = F.relu(self.fc1(o6))
        o8 = F.relu(self.fc2(o7))

        output = self.fc3(o8)
        output = self.fc3(o8)
        return output

    def statistic_pooling(self,x):
        mean_x = x.mean(dim=2)
        std_x = x.std(dim=2)
        mean_std = torch.cat((mean_x, std_x), 1)
        return mean_std


        return x



class CTDNN(nn.Module):
    def __init__(self, in_dim,num_speakers=512):
        super(CTDNN, self).__init__()
        self.frame1_1 = TDNN(input_dim=in_dim, output_dim=512, context_size=3, dilation=1)
        self.frame1_2 = TDNN(input_dim=in_dim, output_dim=512, context_size=5, dilation=1)
        self.frame1_3 = TDNN(input_dim=in_dim, output_dim=512, context_size=9, dilation=1)


        self.frame2 = TDNN(input_dim=512, output_dim=512, context_size=3, dilation=2)
        self.frame3 = TDNN(input_dim=512, output_dim=512, context_size=3, dilation=3)
        #self.frame4 = TDNN(input_dim=512, output_dim=512, context_size=1, dilation=1)
        self.frame5 = TDNN(input_dim=512, output_dim=1500, context_size=1, dilation=1)


        self.fc1 = nn.Linear(5912, 4096)
        self.fc2 = nn.Linear(4096, 2048)
        self.fc3 = nn.Linear(2048, num_speakers)

        self.bn1 = nn.BatchNorm1d(512)
        self.bn2 = nn.BatchNorm1d(512)

    def forward(self, x):
        o1_1 = self.frame1_1(x)
        o1_2 = self.frame1_2(x)
        o1_3 = self.frame1_3(x)

        o2_1 = self.frame2(o1_1)
        o2_2 = self.frame2(o1_2)
        o2_3 = self.frame2(o1_3)

        o3_1 = self.frame3(o2_1)
        o3_2 = self.frame3(o2_2)
        o3_3 = self.frame3(o2_3)

        o5_1=self.frame5(o3_1)
        o5_2=self.frame5(o3_2)
        o5_3=self.frame5(o3_3)

        o6_1 = self.statistic_pooling(o5_1)
        o6_2 = self.statistic_pooling(o5_2)
        o6_3 = self.statistic_pooling(o5_3)

        o7 = torch.cat((o6_1,o6_2,o6_3),1)

        o8 = F.relu(self.fc1(o7))
        o9 = F.relu(self.fc2(o8))

        output = self.fc3(o9)
        return output

    def statistic_pooling(self, x):
        mean_x = x.mean(dim=2)
        std_x = x.std(dim=2)
        mean_std = torch.cat((mean_x, std_x), 1)
        return mean_std

        return x

