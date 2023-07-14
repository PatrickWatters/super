# Imports PyTorch packages.
import torch.nn.functional as F
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler

# Imports other packages.
import  as L
import torch.nn.functional as F

from lightning.pytorch.utilities.model_helpers import get_torchvision_model


class SuperLightningModel(L.LightningModule):
    
    def __init__(self, args, weights=None):
        super().__init__()
        self.arch = args.arch
        self.weights = weights
        self.lr = args.lr
        self.momentum = args.momentum
        self.weight_decay = args.weight_decay
        self.batch_size = args.batch_size
        self.workers = args.workers
        #self.data_path = args.data_path
        #self.index_file_path = args.index_file_path
        self.model = get_torchvision_model(self.arch, weights=self.weights)

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        images, target = batch
        output = self.model(images)
        loss_train = F.cross_entropy(output, target)
        self.log("train_loss", loss_train)
        return loss_train

    def eval_step(self, batch, batch_idx, prefix: str):
        images, target = batch
        output = self.model(images)
        loss_val = F.cross_entropy(output, target)
        self.log(f"{prefix}_loss", loss_val)
        return loss_val

    def validation_step(self, batch, batch_idx):
        return self.eval_step(batch, batch_idx, "val")

    def test_step(self, batch, batch_idx):
        return self.eval_step(batch, batch_idx, "test")

    def configure_optimizers(self):
        optimizer = optim.SGD(self.parameters(), lr=self.lr, momentum=self.momentum, weight_decay=self.weight_decay)
        scheduler = lr_scheduler.LambdaLR(optimizer, lambda epoch: 0.1 ** (epoch // 30))
        return [optimizer], [scheduler]