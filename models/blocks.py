import torch
import torch.nn as nn


def weight_init(m):
    if isinstance(m, nn.Linear):
        m.weight.data.normal_(0, 0.001)
        m.bias.data.zero_()
    if isinstance(m, nn.Conv2d):
        nn.init.xavier_normal_(m.weight.data)
        if m.bias is not None:
            m.bias.data.zero_()
    if isinstance(m, nn.ConvTranspose2d):
        nn.init.xavier_normal_(m.weight.data)
        m.bias.data.zero_()
    if isinstance(m, nn.BatchNorm2d):
        m.weight.data.normal_(0, 0.001)
        m.bias.data.zero_()


class DownSample(nn.Module):
    def __init__(self, in_c, out_c, norm=nn.BatchNorm2d, activ='lrelu'):
        super(DownSample, self).__init__()
        self.conv = nn.Conv2d(
            in_channels=in_c, out_channels=out_c, kernel_size=3, stride=2, padding=1)
        self.norm = norm(out_c)
        if activ == 'lrelu':
            self.lrelu = nn.LeakyReLU(0.1)
        else:
            self.lrelu = nn.ReLU()

    def forward(self, x):
        x = self.conv(x)
        x = self.norm(x)
        x = self.lrelu(x)
        return x


class UpSample(nn.Module):
    def __init__(self, c_in, c_out, norm=nn.BatchNorm2d, activ='lrelu'):
        super(UpSample, self).__init__()

        self.upsample = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            nn.Conv2d(c_in, c_out, kernel_size=3, padding=1)
        )
        self.norm = norm(c_out)
        if activ == 'lrelu':
            self.lrelu = nn.LeakyReLU(0.1)
        else:
            self.lrelu = nn.ReLU()

    def forward(self, x):
        x = self.upsample(x)
        x = self.norm(x)
        x = self.lrelu(x)

        return x


class UpSampleWithSkip(nn.Module):
    def __init__(self, c_in, c_out, norm=nn.BatchNorm2d):
        super(UpSampleWithSkip, self).__init__()

        self.upsample = UpSample(c_in, c_out)

    def forward(self, x, skip):
        x = self.upsample(x)
        x = torch.cat((x, skip), dim=1)

        return x


class conv_block(nn.Module):
    def __init__(self, in_c, out_c, kernel_size=3, stride=1, padding=1, norm=nn.BatchNorm2d, activ='lrelu', dilation=1):
        super(conv_block, self).__init__()
        self.conv = nn.Conv2d(in_channels=in_c, out_channels=out_c,
                              kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation)
        if norm is not None:
            self.norm = norm(out_c)
        else:
            self.norm = None
        if activ == 'lrelu':
            self.lrelu = nn.LeakyReLU(0.1)
        else:
            self.lrelu = nn.ReLU()

    def forward(self, x):
        x = self.conv(x)
        if self.norm is not None:
            x = self.norm(x)
        x = self.lrelu(x)
        return x


class convResBlk(nn.Module):
    def __init__(self, c_in, c_out, norm=nn.BatchNorm2d, activ='lrelu'):
        super(convResBlk, self).__init__()
        c_middle = max(c_in, c_out)

        self.conv1 = nn.Conv2d(
            c_in, c_middle, kernel_size=3, stride=1, padding=1)
        self.norm1 = norm(c_middle)
        if activ == 'lrelu':
            self.relu1 = nn.LeakyReLU(0.1)
        else:
            self.relu1 = nn.ReLU()

        self.conv2 = nn.Conv2d(
            c_middle, c_out, kernel_size=3, stride=1, padding=1)
        self.norm2 = norm(c_out)
        if activ == 'lrelu':
            self.relu = nn.LeakyReLU(0.1)
        else:
            self.relu = nn.ReLU()

        # skip connection
        self.conv3 = nn.Conv2d(c_in, c_out, kernel_size=1, stride=1, padding=0)
        self.norm3 = norm(c_out)

    def forward(self, x):
        h = self.conv1(x)
        h = self.norm1(h)
        h = self.relu1(h)

        h = self.conv2(h)
        h = self.norm2(h)

        # reslink
        x = self.conv3(x)
        x = self.norm3(x)

        x = x + h
        x = self.relu(x)

        return x


class AttrEncoder(nn.Module):
    def __init__(self):
        super(AttrEncoder, self).__init__()
        self.conv_512 = conv_block(
            3, 32, kernel_size=7, padding=3, norm=None, activ='lrelu')
        self.down1 = DownSample(
            32, 64, norm=nn.BatchNorm2d, activ='lrelu')  # -> 64*256*256
        self.down2 = DownSample(
            64, 128, norm=nn.BatchNorm2d, activ='lrelu')  # -> 128*128*128
        self.down3 = DownSample(
            128, 256, norm=nn.BatchNorm2d, activ='lrelu')  # -> 256*64*64
        self.down4 = DownSample(
            256, 512, norm=nn.BatchNorm2d, activ='lrelu')  # -> 512*32*32
        self.down5 = DownSample(
            512, 1024, norm=nn.BatchNorm2d, activ='lrelu')  # -> 512*16*16
        self.convResBlk_16 = convResBlk(
            1024, 1024, norm=nn.BatchNorm2d, activ='lrelu')  # -> 1024*16*16

    def forward(self, x):
        x = self.conv_512(x)
        d1 = self.down1(x)
        d2 = self.down2(d1)
        d3 = self.down3(d2)
        d4 = self.down4(d3)
        d5 = self.down5(d4)
        attr = self.convResBlk_16(d5)
        return attr, d4, d3, d2, d1


class AttrDecoder(nn.Module):
    def __init__(self):
        super(AttrDecoder, self).__init__()
        self.up1 = UpSampleWithSkip(1024, 512)  # -> 512*32*32
        self.up2 = UpSampleWithSkip(1024, 256)  # -> 256*64*64
        self.up3 = UpSampleWithSkip(512, 128)  # -> 128*128*128
        self.up4 = UpSampleWithSkip(256, 64)  # -> 64*256*256
        self.up5 = UpSample(128, 64)  # -> 32*512*512
        self.conv_img = nn.Conv2d(64, 3, kernel_size=3, padding=1)

        self.apply(weight_init)

    def forward(self, attrs):
        attr, d4, d3, d2, d1 = attrs
        x1 = self.up1(attr, d4)
        x2 = self.up2(x1, d3)
        x3 = self.up3(x2, d2)
        x4 = self.up4(x3, d1)
        x5 = self.up5(x4)
        i = self.conv_img(x5)
        out = torch.tanh(i)
        return out
