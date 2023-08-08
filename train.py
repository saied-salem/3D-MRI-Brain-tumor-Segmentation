from monai.metrics import DiceMetric
from monai.transforms import (
    AsDiscrete,
    Activations,
)
from monai import transforms
from monai.data import decollate_batch
from monai.metrics import DiceMetric
import wandb
import logging
import os
import random
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms.functional as TF
from pathlib import Path
from torch import optim
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm
from metrics.evaluate import evaluate

def train_model(
        wb_project_name,
        wb_run_name,
        model,
        loss_func,
        device,
        weights_dir,
        train_loader,
        val_loader,
        input_channels,
        output_classes,
        VAE_param,
        load_weights =True,
        epochs: int = 5,
        batch_size: int = 2,
        learning_rate: float = 1e-5,
        save_checkpoint: bool = True,

        amp: bool = False,
        weight_decay: float = 1e-8,
        momentum: float = 0.999,
        gradient_clipping: float = 1.0,
        
):

    best_metric = -1
    best_metric_epoch = -1
    best_metrics_epochs_and_time = [[], [], []]
    epoch_loss_values = []
    metric_values = []
    metric_values_tc = []
    metric_values_wt = []
    metric_values_et = []

    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
    logging.info(f'Using device {device}')

    # Change here to adapt to your data
    # n_channels=3 for RGB images
    # n_classes is the number of probabilities you want to get per pixe
    model = model.to(memory_format=torch.channels_last)
    print(model.named_parameters())
    logging.info(f'Network:\n'
                 f'\t{input_channels} input channels\n'
                 f'\t{output_classes} output channels (classes)\n'
                #  f'\t{"Bilinear" if model.bilinear else "Transposed conv"} upscaling'
                 )

    if load_weights:
        weights_file= os.listdir(weights_dir)
        weights_path = weights_dir + weights_file[1]
        state_dict = torch.load(weights_path, map_location=device)
        del state_dict['mask_values']
        model.load_state_dict(state_dict)
        logging.info(f'Model loaded from {weights_dir}')


    model.to(device=device)
    # 1. Create dataset
    if output_classes == 1:
        multi_class = False
    else:
        multi_class = True

    n_train = len(train_loader)
    n_val = int(len(val_loader))
    # 3. Create data loaders

    # (Initialize logging)
    experiment = wandb.init(project= wb_project_name, entity="ultra-sound-segmentation" , name = wb_run_name ,resume='allow', anonymous='must')

    experiment.config.update(
        dict(epochs=epochs, 
             batch_size=batch_size, 
             learning_rate=learning_rate,
             val_percent= np.round(n_val/n_val+n_train), 
             save_checkpoint=save_checkpoint, 
             amp=amp)
    )

    logging.info(f'''Starting training:
        Epochs:          {epochs}
        Batch size:      {batch_size}
        Learning rate:   {learning_rate}
        Training size:   {n_train}
        Validation size: {n_val}
        Checkpoints:     {save_checkpoint}
        Device:          {device.type}

        Mixed Precision: {amp}
    ''')

    post_trans = transforms.Compose([Activations(sigmoid=True), AsDiscrete(threshold=0.5)])
    dice_metric = DiceMetric(include_background=True, reduction="mean")
    dice_metric_batch = DiceMetric(include_background=True, reduction="mean_batch")

    # 4. Set up the optimizer, the loss, the learning rate scheduler and the loss scaling for AMP
    optimizer = optim.AdamW(model.parameters(),
                              lr=learning_rate, foreach=True, weight_decay=weight_decay)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'max',factor= 0.8, patience=4)  # goal: maximize Dice score
    grad_scaler = torch.cuda.amp.GradScaler(enabled=amp)

    global_step = 0
    best_metric =-1
    log_table = []
    # 5. Begin training
    for epoch in range(1, epochs + 1):
        model.train()
        epoch_loss = 0
        with tqdm(total=n_train, desc=f'Epoch {epoch}/{epochs}', unit='img') as pbar:
            for batch in train_loader:
                images, true_masks = batch['image'], batch['mask']
                # print()
                # print('true_masks',true_masks.shape)
                # print('true_masks',torch.unique(true_masks))

                assert images.shape[1] == input_channels, \
                    f'Network has been defined with {input_channels} input channels, ' \
                    f'but loaded images have {images.shape[1]} channels. Please check that ' \
                    'the images are loaded correctly.'
                
                with torch.autocast(device.type if device.type != 'mps' else 'cpu', enabled=amp):

                    if VAE_param:
                        masks_pred, VAE_loss  = model(images)
                        loss = loss_func(masks_pred, true_masks) +VAE_loss
                    else:
                        masks_pred = model(images)
                        loss = loss_func(masks_pred, true_masks)

                optimizer.zero_grad(set_to_none=True)
                grad_scaler.scale(loss).backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clipping)
                grad_scaler.step(optimizer)
                grad_scaler.update()

                pbar.update(images.shape[0])
                global_step += 1
                epoch_loss += loss.item()
                experiment.log({
                    'train loss': loss.item(),
                    'step': global_step,
                    'epoch': epoch
                })
                pbar.set_postfix(**{'loss (batch)': loss.item()})
                # log_table.append({'image':  wandb.Image(images[0].cpu()), 
                #                   'predictied_mask': wandb.Image(masks_pred.argmax(dim=1)[0].float().cpu()), 
                #                   "true_mask":  wandb.Image(true_masks[0].float().cpu())})
                # log_df = pd.DataFrame(log_table)

                # Evaluation round
                # division_step = (n_train // (5 * batch_size))
                # if division_step > 0:
                #     if global_step % division_step == 0:
            histograms = {}
            for tag, value in model.named_parameters():
                tag = tag.replace('/', '.')
                if not (torch.isinf(value) | torch.isnan(value)).any():
                    histograms['Weights/' + tag] = wandb.Histogram(value.data.cpu())
                if value.grad==None:
                  # print("LOoooooooooooooooooooooool")
                  pass
                # print("vaaaaaaal",value.grad)
                else:
                  if not ( torch.isnan(value.grad) | torch.isinf(value.grad) ).any():
                      histograms['Gradients/' + tag] = wandb.Histogram(value.grad.data.cpu())

            
            evaluate(model, val_loader,dice_metric, dice_metric_batch, device, amp, VAE_param)
            curr_metric = dice_metric.aggregate().item()
            scheduler.step(curr_metric)

            metric_values.append(curr_metric)
            metric_batch = dice_metric_batch.aggregate()
            metric_tc = metric_batch[0].item()
            metric_values_tc.append(metric_tc)
            metric_wt = metric_batch[1].item()
            metric_values_wt.append(metric_wt)
            metric_et = metric_batch[2].item()
            metric_values_et.append(metric_et)
            dice_metric.reset()
            dice_metric_batch.reset()

            
            logging.info('Validation Dice score: {}'.format(curr_metric))
            if curr_metric>best_metric :
                Path(weights_dir).mkdir(parents=True, exist_ok=True)
                state_dict = model.state_dict()
                state_dict['mask_values'] = train_set.mask_values
                saving_path = weights_dir + 'best_checkpoint_dice_val_score.pth'
                torch.save(state_dict, saving_path)
                logging.info(f'Checkpoint {epoch} saved!')
                best_metric = curr_metric
            try:
                experiment.log({
                    'learning rate': optimizer.param_groups[0]['lr'],
                    'validation Dice': curr_metric,
                    'validation Dice_tc': metric_tc,
                    'validation Dice_wt': metric_wt,
                    'validation Dice_et': metric_et,
                    'step': global_step,
                    'epoch': epoch,
                    **histograms
                })
            except:
                pass