import torch
import torch.nn.functional as F
from lightning.pytorch.callbacks import Callback

class CrossEntropyLossWithReshape(torch.nn.Module):
    def __init__(self, ignore_index=-100, label_smoothing=0.0):
        super().__init__()
        self.ignore_index = ignore_index
        self.label_smoothing = label_smoothing

    def forward(self, pred, target):
        """
        pred - (D1, D2, ..., Dn, C)
        target - (D1, D2, ..., Dn)
        C - number of classes
        """
        pred_flat = pred.view(-1, pred.shape[-1])  # D1*D2*...*Dn x C
        target_flat = target.reshape(-1)  # D1*D2*...*Dn
        return F.cross_entropy(pred_flat,
                            target_flat,
                            ignore_index=self.ignore_index,
                            label_smoothing=self.label_smoothing)



class EmptyCudaCacheCallback(Callback):
    def on_train_epoch_end(self, trainer, pl_module):
        torch.cuda.empty_cache()

