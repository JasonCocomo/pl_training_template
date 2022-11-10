
import torch
import torch.nn.functional as F
import pytorch_lightning as pl
from models.rpnet import RepairNet


class RPCore(pl.LightningModule):
    def __init__(self, *args, **kwargs):
        super().__init__()
        self.learning_rate = kwargs['learning_rate']
        rpnet_config = kwargs['rpnet_config']
        self.model = RepairNet(rpnet_config)

    def training_step(self, batch, batch_idx):
        fake_imgs, gt_imgs = batch['fake_img'], batch['gt_img']
        fake_imgs = fake_imgs.to(self.device)
        gt_imgs = gt_imgs.to(self.device)
        pred_imgs = self.model(fake_imgs)
        loss = F.mse_loss(pred_imgs, gt_imgs)
        loss_dict = {
            'mse': loss
        }
        self.log_dict(loss_dict, prog_bar=True,
                      logger=True, on_step=True, on_epoch=True)
        self.log("global_step", self.global_step,
                 prog_bar=True, logger=True, on_step=True, on_epoch=False)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        return optimizer
