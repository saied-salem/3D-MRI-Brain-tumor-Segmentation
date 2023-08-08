from train import train_model
from utils.data_loader import getLoader
from train import train_model

import torch 
from monai.networks.nets import SegResNetVAE
from monai.losses import DiceLoss
import monai 
import json


configerations= None
with open('model_config.json') as json_file:
    configerations = json.load(json_file)
    # Print the type of data variable
    print("Type:", type(configerations))
    print(configerations)

json_path = '/content/drive/MyDrive/Brain_tumor_segmentation/brain_dataset.json'
train_loader , val_loader= getLoader(configerations['batch_size'],
                                     configerations['val_percent'],
                                     json_path)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = SegResNetVAE(
        input_image_size = (128, 128, 64),
        vae_estimate_std = True,
        vae_default_std = 0.3,
        vae_nz = 256,
        spatial_dims = 3,
        init_filters = 8,
        in_channels = configerations['input_channels'],
        out_channels = configerations['output_classes'],
        dropout_prob = None,
        act = 'RELU',
        norm = ('GROUP', {'num_groups': 8}),
        use_conv_final = True,
        blocks_down = (1, 2, 2, 4),
        blocks_up = (1, 1, 1),
        upsample_mode = monai.utils.enums.UpsampleMode.NONTRAINABLE,
            ).to(device)

loss_func = DiceLoss(smooth_nr=0, smooth_dr=1e-5, squared_pred=True, to_onehot_y=False, sigmoid=True)

if __name__ == '__main__':

    train_model(
        wb_project_name = configerations['wb_project_name'],
        wb_run_name = configerations['wb_run_name'],
        model = model,
        loss_func= loss_func
        device = device,
        load_weights = configerations['load_weights'],
        weights_dir = configerations['saving_weights_dir'],
        train_loader =train_loader ,
        val_loader= val_loader,
        input_channels = configerations['input_channels'],
        output_classes = configerations['output_classes'],
        epochs = configerations['epochs'],
        batch_size = configerations['batch_size'],
        learning_rate = configerations['learning_rate'],
        save_checkpoint = True,
        VAE_param = configerations['VAE'],
        )