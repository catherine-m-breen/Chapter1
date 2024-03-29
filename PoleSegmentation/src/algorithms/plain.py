'''
Cat Updates:
- adjusted outputs and inputs to both be floats
- Updated to track MeanIoU, because in utils.py it created a dictionary to track it
- in validation step, added a line to get the right dimension of the labels, not just predicted values. 
    Selected the positive axis (should make sure it shouldnt be 0). 

'''

import time
import os
import numpy as np
from PIL import Image

import pytorch_lightning as pl # cat switched order
import torch
import torch.optim as optim


from .utils import StreamSegMetrics, PolyLR, denormalize, snowpole_cmap
from src import models
import IPython

__all__ = [
    'Plain'
]

class Plain(pl.LightningModule):

    """
    Overall training function.
    """

    name = 'Plain'

    def __init__(self, conf, **kwargs):
        super().__init__()
        self.hparams.update(conf.__dict__)
        self.save_hyperparameters(ignore=['conf', 'train_class_counts'])
        self.net = models.__dict__[self.hparams.model_name](num_cls=self.hparams.num_classes, 
                                                            output_stride=self.hparams.output_stride)
                             
        self.metrics = StreamSegMetrics(self.hparams.num_classes)

    def configure_optimizers(self):
        net_optim_params_list = [
            {'params': self.net.feature.parameters(),
             'lr': self.hparams.lr_feature * torch.cuda.device_count(),
             'momentum': self.hparams.momentum_feature,
             'weight_decay': self.hparams.weight_decay_feature},
            {'params': self.net.classifier.parameters(),
             'lr': self.hparams.lr_classifier * torch.cuda.device_count(),
             'momentum': self.hparams.momentum_classifier,
             'weight_decay': self.hparams.weight_decay_classifier}
        ]
        # Setup optimizer and optimizer scheduler
        optimizer = torch.optim.SGD(net_optim_params_list)
        scheduler = PolyLR(optimizer, self.hparams.num_iters, power=0.9)
        return [optimizer], [{'scheduler': scheduler, 'interval': 'step', 'frequency': 1}]

    def on_train_start(self):
        self.net.setup_criteria()

    def training_step(self, batch, batch_idx):
        data, labels = batch
        outputs = self.net(data)
        loss = self.net.criterion_seg(outputs, labels.type(torch.float32)) ## cat edit: changed to float 
        self.log("train_loss", loss)
        # IPython.embed()
        ## get the valid_mean_IoU
        scores = self.metrics.get_results()
        self.log('valid_mean_IoU', scores['Mean IoU']) ## cat edit: added tracking of valid_mean_IoU
        #self.log_image()

        return loss

    def on_validation_start(self):
        self.metrics.reset()

    def validation_step(self, batch, batch_idx):
        data, labels = batch
        outputs = self.net(data)
        #IPython.embed()
        labels = labels.detach().max(dim=1)[1]  ## cat addition, just channel for snow [1]
        preds = outputs.detach().max(dim=1)[1] 
        self.metrics.update(labels.cpu().numpy(), preds.cpu().numpy())

        return (data.detach().cpu().numpy(), 
                preds.detach().cpu().numpy(),
                labels.detach().cpu().numpy())

    def on_validation_epoch_end(self, outputs): ## instead of validation_epoch_end #on_validation_epoch_end
        #IPython.embed()
        scores = self.metrics.get_results()
        self.log('valid_mean_IoU', scores['Mean IoU'])
        self.log('valid_overall_acc', scores['Overall Acc'])
        self.log('valid_mean_acc', scores['Mean Acc'])

        total_data = np.concatenate([x[0] for x in outputs], axis=0)
        total_preds = np.concatenate([x[1] for x in outputs], axis=0)
        total_labels = np.concatenate([x[2] for x in outputs], axis=0)
        
        for i in np.random.choice(len(total_labels), 10, replace=False):
            image = total_data[i]
            target = total_labels[i]
            pred = total_preds[i]

            image = (denormalize(image, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]) * 255).transpose(1, 2, 0).astype(np.uint8)
            target = snowpole_cmap()[target].astype(np.uint8)
            pred = snowpole_cmap()[pred].astype(np.uint8)

            image = Image.fromarray(image)
            target = Image.fromarray(target)
            pred = Image.fromarray(pred)

            if self.logger.__class__.__name__ == 'CometLogger':
                self.logger.experiment.log_image(image, name='{}_data.png'.format(i), overwrite=True)
                self.logger.experiment.log_image(target, name='{}_target.png'.format(i), overwrite=True)
                self.logger.experiment.log_image(pred, name='{}_pred.png'.format(i), overwrite=True)
            else:
                if not os.path.exists('./log/temp_imgs'):  
                     os.makedirs('./log/temp_imgs')
                image.save('./log/temp_imgs/{}_data.png'.format(i))
                target.save('./log/temp_imgs/{}_target.png'.format(i))
                pred.save('./log/temp_imgs/{}_pred.png'.format(i))
