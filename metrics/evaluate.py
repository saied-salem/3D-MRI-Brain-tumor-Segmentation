import torch
import torch.nn.functional as F
from tqdm import tqdm

from monai.metrics import DiceMetric
from monai.transforms import (
    AsDiscrete,
    Activations,
)
from monai import transforms
from monai.data import decollate_batch
from monai.metrics import DiceMetric

@torch.inference_mode()
def evaluate(model, dataloader,metric,metric_batch, post_trans, device, amp , VAE_param =False,):

    

    model.eval()
    num_val_batches = len(dataloader)
    dice_score = 0

    # iterate over the validation set
    with torch.autocast(device.type if device.type != 'mps' else 'cpu', enabled=amp):
        for batch in tqdm(dataloader, total=num_val_batches, desc='Validation round', unit='batch', leave=False):
            val_inputs, val_labels = (
                dataloader["images"].to(device),
                dataloader["mask"].to(device),
            )
            # val_outputs = inference(val_inputs)
            # print(val_inputs.shape)

            if VAE_param:
                val_outputs, loss = model(val_inputs)
            else:
                val_outputs= model(val_inputs)

            # print(val_outputs)
            val_outputs = [post_trans(i) for i in decollate_batch(val_outputs)]
            metric(y_pred=val_outputs , y=val_labels)
            metric_batch(y_pred=val_outputs, y=val_labels)

    model.train()
    # return dice_score / max(num_val_batches, 1)