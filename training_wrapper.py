import torch
import pytorch_lightning as pl
import torch.nn.functional as F
from torchmetrics import Accuracy

class TrainingWrapper(pl.LightningModule):
    def __init__(self, model, lr=0.05):
        super().__init__()
        
        self.model = model
        self.accuracy = Accuracy()
        self.save_hyperparameters()

    def training_step(self, batch, batch_idx):
        x, y = batch
        outputs = self.model(x)
        loss = F.cross_entropy(outputs.squeeze(), y.squeeze())
        self.log("train_loss", loss)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        outputs = self.model(x)
        loss = F.cross_entropy(outputs.squeeze(), y.squeeze())
        probs = F.softmax(outputs, dim=1)
        preds = torch.argmax(probs, dim=1)
        acc = self.accuracy(preds, y)
        metrics = {'val_acc': acc, 'val_loss': loss}
        self.log_dict(metrics)
        
    def test_step(self, batch, batch_idx):
        x, y = batch
        outputs = self.model(x)
        loss = F.cross_entropy(outputs.squeeze(), y.squeeze())
        probs = F.softmax(outputs, dim=1)
        preds = torch.argmax(probs, dim=1)
        acc = self.accuracy(preds, y)
        metrics = {'test_acc': acc, 'test_loss': loss}
        self.log_dict(metrics)

    def configure_optimizers(self):
        optimizer = torch.optim.SGD(
            self.model.parameters(),
            lr=self.hparams.lr,
            momentum=0.9,
            weight_decay=5e-4
        )
        return optimizer