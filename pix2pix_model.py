import torch
import torch.nn as nn

class DownSample(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, apply_batchnorm=True):
        super(DownSample, self).__init__()
        layers = []
        
        conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride=2, padding=1, bias=False)
        torch.nn.init.normal_(conv.weight.data, 0.0, 0.02)
        layers.append(conv)
        
        if apply_batchnorm:
            layers.append(nn.BatchNorm2d(out_channels))
        
        '''
        negative_slope: keras default is 0.3 but
        pytorch default is 0.1
        '''
        layers.append(nn.LeakyReLU(0.3)) 
        
        self.model = nn.Sequential(*layers)
        
    def forward(self, x):
        x = self.model(x)
        return x
        
class UpSample(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, apply_dropout=False):
        super(UpSample, self).__init__()
        layers = []
        
        conv_transpose = nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride=2, padding=1, bias=False)
        torch.nn.init.normal_(conv_transpose.weight.data, 0.0, 0.02)
        layers.append(conv_transpose)
        
        layers.append(nn.BatchNorm2d(out_channels))
        if apply_dropout:
            layers.append(nn.Dropout(0.5))
        layers.append(nn.ReLU()) # What is inplace?
        
        self.model = nn.Sequential(*layers)
        
    def forward(self, x, skip_input):
        x = self.model(x)
        x = torch.cat((x, skip_input), dim=1)
        
        return x
        
class Generator(nn.Module):
    def __init__(self, in_channels=4, out_channels=4):
        super(Generator, self).__init__()
        
        self.down1 = DownSample(in_channels, 64, kernel_size=4, apply_batchnorm=False) # (batch_size, 64, 256, 256)
        self.down2 = DownSample(64, 128, 4) # (batch_size, 128, 128, 128)
        self.down3 = DownSample(128, 256, 4) # (batch_size, 256, 64, 64)
        self.down4 = DownSample(256, 512, 4) # (batch_size, 512, 32, 32)
        self.down5 = DownSample(512, 512, 4) # (batch_size, 512, 16, 16)
        self.down6 = DownSample(512, 512, 4) # (batch_size, 512, 8, 8)
        self.down7 = DownSample(512, 512, 4) # (batch_size, 512, 4, 4)
        self.down8 = DownSample(512, 512, 4) # (batch_size, 512, 2, 2)
        
        self.up1 = UpSample(512, 512, 4, apply_dropout=True) # (batch_size, 1024, 4, 4)
        self.up2 = UpSample(1024, 512, 4, apply_dropout=True) # (batch_size, 1024, 8, 8)
        self.up3 = UpSample(1024, 512, 4, apply_dropout=True) # (batch_size, 1024, 16, 16)
        self.up4 = UpSample(1024, 512, 4) # (batch_size, 1024, 32, 32)
        self.up5 = UpSample(1024, 256, 4) # (batch_size, 1024, 64, 64)
        self.up6 = UpSample(512, 128, 4) # (batch_size, 1024, 128, 128)
        self.up7 = UpSample(256, 64, 4) # (batch_size, 1024, 256, 256)
        
        layers = []
        conv_transpose = nn.ConvTranspose2d(128, out_channels, kernel_size=4, stride=2, padding=1)
        torch.nn.init.normal_(conv_transpose.weight.data, 0.0, 0.02)
        layers.append(conv_transpose)
        layers.append(nn.Tanh())
        
        self.final = nn.Sequential(*layers)
        
    def forward(self, x):
        # U-Net generator with skip connections from encoder to decoder
        d1 = self.down1(x)
        d2 = self.down2(d1)
        d3 = self.down3(d2)
        d4 = self.down4(d3)
        d5 = self.down5(d4)
        d6 = self.down6(d5)
        d7 = self.down7(d6)
        d8 = self.down8(d7)
        u1 = self.up1(d8, d7)
        u2 = self.up2(u1, d6)
        u3 = self.up3(u2, d5)
        u4 = self.up4(u3, d4)
        u5 = self.up5(u4, d3)
        u6 = self.up6(u5, d2)
        u7 = self.up7(u6, d1)
        
        result = self.final(u7)
        return result
    
class Discriminator(nn.Module):
    def __init__(self, in_channels=4):
        super(Discriminator, self).__init__()
        
        self.down1 = DownSample(in_channels * 2, 64, 4, False)
        self.down2 = DownSample(64, 128, 4)
        self.down3 = DownSample(128, 256, 4)
        
        layers = []
        layers.append(nn.ZeroPad2d(padding=1))
        conv1 = nn.Conv2d(256, 512, 4, 1)
        torch.nn.init.normal_(conv1.weight.data, 0.0, 0.02)
        layers.append(conv1)
        layers.append(nn.BatchNorm2d(512))
        layers.append(nn.LeakyReLU(0.3))
        layers.append(nn.ZeroPad2d(padding=1))
        conv2 = nn.Conv2d(512, 1, 4, 1)
        torch.nn.init.normal_(conv2.weight.data, 0.0, 0.02)
        layers.append(conv2)
        
        self.final = nn.Sequential(*layers)
        
    def forward(self, input_image, real_image):
        x = torch.cat([input_image, real_image], dim=1)
        x = self.down1(x)
        x = self.down2(x)
        x = self.down3(x)
        x = self.final(x)
        return x