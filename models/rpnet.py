from .blocks import AttrDecoder, AttrEncoder
import torch
from torch import nn


class RepairNet(nn.Module):
    def __init__(self, *args, **kwargs):
        super(RepairNet, self).__init__()
        self.encoder = AttrEncoder()
        self.decoder = AttrDecoder()

    def forward(self, imgs):
        attrs = self.encoder(imgs)
        outs = self.decoder(attrs)
        return outs


if __name__ == '__main__':
    model = RepairNet()
    imgs = torch.rand((4, 3, 512, 512), dtype=torch.float32)
    outs = model(imgs)
