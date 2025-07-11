
import torch
import torch.nn as nn
import torch.nn.functional as F
from fed_attack.attack import MODEL
from torchvision.models import resnet18
class CNN(nn.Module):
    def __init__(self):
        super().__init__()

        # ---- Feature extractor ----
        self.features = nn.Sequential(
            # Block 1
            nn.Conv2d(3, 32, kernel_size=3, padding=1, bias=False),  # same padding
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(32),

            nn.Conv2d(32, 32, kernel_size=3, padding=1, bias=False),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(32),

            nn.MaxPool2d(2),               # 32 -> 16
            nn.Dropout(0.25),

            # Block 2
            nn.Conv2d(32, 64, kernel_size=3, padding=1, bias=False),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(64),

            nn.Conv2d(64, 64, kernel_size=3, padding=1, bias=False),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(64),

            nn.MaxPool2d(2),               # 16 -> 8
            nn.Dropout(0.25),

            # Block 3
            nn.Conv2d(64, 128, kernel_size=3, padding=1, bias=False),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(128),

            nn.Conv2d(128, 128, kernel_size=3, padding=1, bias=False),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(128),

            nn.MaxPool2d(2),               # 8 -> 4
            nn.Dropout(0.25),
        )

        # ---- Classifier ----
        self.classifier = nn.Sequential(
            nn.Flatten(),                       # 128 × 4 × 4 = 2048
            nn.Linear(128 * 4 * 4, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.25),
            nn.Linear(128, 10),
            nn.Softmax(dim=1)                   # giữ nguyên như Keras; 
                                                # nếu dùng CrossEntropyLoss thì bỏ Softmax
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x

class Resnet18(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = resnet18()
        self.model.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.model.maxpool = nn.Identity()
        self.model.fc = nn.Linear(self.model.fc.in_features, 10)

    def forward(self, x):
        return self.model(x)

class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        self.latent_vector_size = 128
        self.deconv1 = nn.ConvTranspose2d( self.latent_vector_size, 64 * 8, 4, 1, 0, bias=False)
        self.norm1 = nn.BatchNorm2d(64 * 8)
        self.relu = nn.ReLU(True)
        self.deconv2 = nn.ConvTranspose2d(64 * 8, 64 * 4, 4, 2, 1, bias=False)
        self.norm2 = nn.BatchNorm2d(64 * 4)
        self.deconv3 = nn.ConvTranspose2d( 64 * 4, 64 * 2, 4, 2, 1, bias=False)
        self.norm3 = nn.BatchNorm2d(64 * 2)
        
        self.deconv4 = nn.ConvTranspose2d( 64 * 2, 3, 4, 2, 1, bias=False)
        self.tanh = nn.Tanh()
        

    def decode(self, z):
        
        z = self.relu(self.norm1(self.deconv1(z))) # b, 16, 5, 5
        z = self.relu(self.norm2(self.deconv2(z))) # b, 8, 15, 15
        z = self.relu(self.norm3(self.deconv3(z))) # b, 1, 28, 28
        z = self.tanh(self.deconv4(z))
        return z

    def forward(self, z):
        return self.decode(z)


class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()

        self.conv1 = nn.Conv2d(3, 64, 4, 2, 1, bias=False)
        self.lrelu = nn.LeakyReLU(0.2, inplace=True)
        self.conv2 = nn.Conv2d(64, 64 * 2, 4, 2, 1, bias=False)
        self.norm2 = nn.BatchNorm2d(64 * 2)
        self.conv3 = nn.Conv2d(64 * 2, 64 * 4, 4, 2, 1, bias=False)
        self.norm3 = nn.BatchNorm2d(64 * 4)
        self.conv4 = nn.Conv2d(64 * 4, 1, 5, 2, 1, bias=False)
        self.sigmoid = nn.Sigmoid()

        
    def discriminator(self, x):

        x = self.lrelu(self.conv1(x))
        x = self.lrelu(self.norm2(self.conv2(x)))
        x = self.lrelu(self.norm3(self.conv3(x)))
        x = self.conv4(x)
        return self.sigmoid(x)

    def forward(self, x):
        out = self.discriminator(x)
        return out.view(-1, 1).squeeze(1)
    
def weights_init_normal(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)
        
        
def get_model(config):
    if config == "cnn": 
        return CNN
    elif config == "resnet18": 
        return Resnet18
    else: 
        raise ValueError(f"Unknown model name: {config}")

Net = get_model(MODEL)