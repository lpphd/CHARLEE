from torch import nn
import torch.nn.functional as F
from tsai.models.InceptionTime import InceptionTime
from tsai.models.ResNet import ResNet


class InceptionTimeClassifier(nn.Module):
    def __init__(self, model_config, data_config):
        super(InceptionTimeClassifier, self).__init__()
        self.c_in = data_config['channels']
        self.c_out = data_config['n_classes']
        self.seq_len = data_config['timesteps']
        self.depth = model_config['inception_depth']
        self.inct = InceptionTime(self.c_in, c_out=self.c_out, seq_len=self.seq_len, depth=self.depth)
        self.input_dropout = nn.Dropout(p=model_config['classifier_input_dropout_perc'])
        layers = []
        if model_config['classifier_use_input_dropout']:
            layers.append(self.input_dropout)
        layers.append(self.inct)
        self.pipeline = nn.Sequential(*layers)

    def forward(self, X):
        x = self.pipeline(X)
        return x

    def computeLoss(self, logits, labels):
        Lacc = F.cross_entropy(logits, labels)
        return Lacc


class ResNetClassifier(nn.Module):
    def __init__(self, model_config, data_config):
        super(ResNetClassifier, self).__init__()
        self.c_in = data_config['channels']
        self.c_out = data_config['n_classes']
        self.resnet = ResNet(self.c_in, self.c_out)
        self.input_dropout = nn.Dropout(p=model_config['classifier_input_dropout_perc'])
        layers = []
        if model_config['classifier_use_input_dropout']:
            layers.append(self.input_dropout)
        layers.append(self.resnet)
        self.pipeline = nn.Sequential(*layers)

    def forward(self, X):
        x = self.pipeline(X)
        return x

    def computeLoss(self, logits, labels):
        Lacc = F.cross_entropy(logits, labels)
        return Lacc
