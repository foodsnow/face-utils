import cv2
import numpy
from torch import nn
import torch.nn.functional as F
import torch

from src.models.utils import process_iin_image, get_iin_from_document


class SimpleDigitModel(nn.Module):
    def __init__(self):
        super(SimpleDigitModel, self).__init__()
        self.conv1 = nn.Conv2d(1, 16, kernel_size=3)
        self.maxPool1 = nn.MaxPool2d(2)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3)
        self.maxPool2 = nn.MaxPool2d(2)
        self.conv2_drop = nn.Dropout2d()
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3)
        self.fc1 = nn.Linear(576, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = self.maxPool1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = self.maxPool2(x)
        x = F.dropout(x, training=self.training)
        x = F.relu(x)
        x = self.conv3(x)
        x = F.relu(x)
        x = x.view(-1, 576)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return F.log_softmax(x)


class InnChecker:

    def __init__(self, path="src/models/checkpoints/simple_digit_model.pt"):
        self.model = SimpleDigitModel()
        checkpoint = torch.load(path)
        self.model.load_state_dict(checkpoint['model_state_dict'])

    def get_iin(self, image):
        self.model.eval()
        opencv_image = cv2.cvtColor(numpy.array(image), cv2.COLOR_RGB2BGR)
        iin_image = get_iin_from_document(opencv_image)
        digits = process_iin_image(self.model, iin_image)
        return "".join([str(i[0]) for i in digits])
