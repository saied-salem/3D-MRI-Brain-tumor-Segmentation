from monai import transforms
from monai import data
from monai.data import decollate_batch
import monai
import json
import random
from sklearn.model_selection import train_test_split

def readDate(json_path ):
  with open(json_path ) as json_file:
    data_set = json.load(json_file)
    # print(data_set)

  return data_set['training'], data_set['val']

def splitData(images_mask_dict, test_size):
    data_size = len(images_mask_dict)
    indices = list(range(data_size))
    train_indices, test_indices = train_test_split(indices, test_size=test_size, random_state=42)
    train_paths = [
        {
        "images": images_mask_dict[i]["images"],
        "mask": images_mask_dict[i]["mask"] ,
        }
        for i in train_indices
        ]

    val_paths =[
        {
        "images": images_mask_dict[i]["images"],
        "mask": images_mask_dict[i]["mask"] ,
        }
        for i in test_indices
        ]

    return train_paths , val_paths

def getLoader(batch_size,val_percent, json_path):
    train_files, validation_files = readDate(json_path)

    train_paths , val_paths= splitData(train_files, val_percent)
    print("train_paths:",len(train_paths))
    train_transform = transforms.Compose(
        [
            transforms.LoadImaged(keys=["images", "mask"]),
            transforms.ConvertToMultiChannelBasedOnBratsClassesd(keys="mask"),
            # transforms.CropForegroundd(
            #     keys=["images", "mask"],
            #     source_key="images",

            # ),
            transforms.RandSpatialCropd(keys=["images", "mask"], roi_size=[128, 128, 64], random_size=False),
            # transforms.RandFlipd(keys=["images", "mask"], prob=0.5, spatial_axis=0),
            # transforms.RandFlipd(keys=["images", "mask"], prob=0.5, spatial_axis=1),
            # transforms.RandFlipd(keys=["images", "mask"], prob=0.5, spatial_axis=2),
            transforms.NormalizeIntensityd(keys="images", nonzero=True, channel_wise=True),
            # transforms.RandScaleIntensityd(keys="images", factors=0.1, prob=1.0),
            # transforms.RandShiftIntensityd(keys="images", offsets=0.1, prob=1.0),
        ]
    )
    val_transform = transforms.Compose(
        [
            transforms.LoadImaged(keys=["images", "mask"]),
            transforms.ConvertToMultiChannelBasedOnBratsClassesd(keys="mask"),
            transforms.NormalizeIntensityd(keys="images", nonzero=True, channel_wise=True),
        ]
    )

    train_ds = data.Dataset(data=train_paths, transform=train_transform)

    train_loader = data.DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=2,
        pin_memory=True,
    )
    val_ds = data.Dataset(data=val_paths, transform=val_transform)
    val_loader = data.DataLoader(
        val_ds,
        batch_size=1,
        shuffle=False,
        num_workers=2,
        pin_memory=True,
    )

    return train_loader, val_loader