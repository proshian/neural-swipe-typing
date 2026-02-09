import torch
from lightning import LightningModule
import torchmetrics

from metrics import get_word_level_accuracy


# ! Make sure:
# * Add metrics

#! Maybe store:
# * batch_size
# * early_stopping_patience

#! Maybe:
# * Checpointing by condition: if model improved on val_loss and val_loss < max_val_loss_to_save


class LitNeuroswipeModel(LightningModule):
    def __init__(self, 
                 model: torch.nn.Module,
                 criterion: torch.nn.Module,
                 word_pad_idx: int,
                 num_classes: int,
                 optimizer_ctor,
                 train_batch_size: int = None,  # to be able to know batch size from checkpoint
                 lr_scheduler_ctor=None,
                 ) -> None:
        super().__init__()

        self.save_hyperparameters(ignore = ["criterion", 'lr_scheduler_ctor', 'optimizer_ctor'])

        self.train_batch_size = train_batch_size

        self.optimizer_ctor = optimizer_ctor
        self.lr_scheduler_ctor = lr_scheduler_ctor

        self.model = model
        self.model.train()
        self.criterion = criterion
        self.word_pad_idx = word_pad_idx

        self.train_token_acc = torchmetrics.classification.Accuracy(
            task="multiclass", num_classes=num_classes, ignore_index=word_pad_idx)
        self.val_token_acc = torchmetrics.classification.Accuracy(
            task="multiclass", num_classes=num_classes, ignore_index=word_pad_idx)
        self.train_token_f1 = torchmetrics.classification.F1Score(
            task="multiclass", num_classes=num_classes, ignore_index=word_pad_idx)
        self.val_token_f1 = torchmetrics.classification.F1Score(
            task="multiclass", num_classes=num_classes, ignore_index=word_pad_idx)

    def forward(self, encoder_in, y, encoder_in_pad_mask, y_pad_mask):
        return self.model.forward(encoder_in, y, encoder_in_pad_mask, y_pad_mask)

    def configure_optimizers(self):
        optimizer = self.optimizer_ctor(self.named_parameters())

        optimizers_configuration = {'optimizer': optimizer}

        if self.lr_scheduler_ctor:
            lr_scheduler = self.lr_scheduler_ctor(optimizer)
            optimizers_configuration['lr_scheduler'] = lr_scheduler
            optimizers_configuration['monitor'] = 'val_loss'

        return optimizers_configuration

    def on_train_epoch_start(self):
        optimizer: torch.optim.Optimizer = self.optimizers()
        lr = optimizer.param_groups[0]['lr']
        self.log('lr', lr, on_step=False, on_epoch=True, logger=True)
        # It's supposed that there's only one param group. 
        # Alternative:
        # for i, group in enumerate(optimizer.param_groups):
        # self.log(f'lr_group_{i}', group['lr'], on_step=False, on_epoch=True, logger=True)

    def training_step(self, batch, batch_idx):
        batch_x, batch_y = batch

        batch_size = batch_y.shape[-1]

        # batch_x, batch_y = move_all_to_device(batch_x, batch_y, self.device)

        # * batch_x is a Tuple of (curve_traj_feats, curve_kb_tokens,
        #   decoder_in, swipe_pad_mask, dec_seq_pad_mask).
        # * batch_y is decoder_out.

        # preds.shape = (chars_seq_len, batch_size, n_classes)

        encoder_in, decoder_in, swipe_pad_mask, dec_seq_pad_mask = batch_x

        pred = self.forward(*batch_x)

        loss = self.criterion(pred, batch_y)


        argmax_pred = torch.argmax(pred, dim=2)
        wl_acccuracy = get_word_level_accuracy(
            argmax_pred.T, batch_y.T, pad_token = self.word_pad_idx, mask = dec_seq_pad_mask)


        flat_y = batch_y.reshape(-1)
        n_classes = pred.shape[-1]
        flat_preds = pred.reshape(-1, n_classes)

        self.train_token_acc(flat_preds, flat_y)
        self.log('train_token_level_accuracy', self.train_token_acc, on_step=True, on_epoch=False)

        self.train_token_f1(flat_preds, flat_y)
        self.log('train_token_level_f1', self.train_token_f1, on_step=True, on_epoch=False)


        self.log("train_word_level_accuracy", wl_acccuracy, on_step=True, on_epoch=True,
                 prog_bar=True, logger=True, batch_size = batch_size)

        self.log("train_loss", loss, on_step=True, on_epoch=True,
                 prog_bar=True, logger=True, batch_size = batch_size)

        return loss

    def validation_step(self, batch, batch_idx):
        batch_x, batch_y = batch
        batch_size = batch_y.shape[-1]
        # batch_x, batch_y = move_all_to_device(batch_x, batch_y, self.device)
        encoder_in, decoder_in, swipe_pad_mask, dec_seq_pad_mask = batch_x
        pred = self.forward(*batch_x)
        loss = self.criterion(pred, batch_y)
        argmax_pred = torch.argmax(pred, dim=2)
        wl_acccuracy = get_word_level_accuracy(
            argmax_pred.T, batch_y.T, pad_token = self.word_pad_idx, mask = dec_seq_pad_mask)


        flat_y = batch_y.reshape(-1)
        n_classes = pred.shape[-1]
        flat_preds = pred.reshape(-1, n_classes)


        self.val_token_acc(flat_preds, flat_y)
        self.log('val_token_level_accuracy', self.train_token_acc, on_step=False, on_epoch=True)

        self.val_token_f1(flat_preds, flat_y)
        self.log('val_token_level_f1', self.train_token_f1, on_step=False, on_epoch=True)



        self.log("val_word_level_accuracy", wl_acccuracy, on_step=False, on_epoch=True,
                 prog_bar=True, logger=True, batch_size = batch_size)
        self.log("val_loss", loss, on_step=False, on_epoch=True, prog_bar=True,
                 logger=True, batch_size = batch_size)
        return loss
