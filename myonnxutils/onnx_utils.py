import numpy as np
import pdb
import pandas as pd
from   tqdm import tqdm 
import matplotlib.pyplot as plt
from   torchvision import transforms 
from   PIL import Image 
import glob, os

import onnxruntime.training.onnxblock as onnxblock
import onnx


def list2dataframe(datetime_list, Irad_list, Iclr_list, Transformed_images_list):
    dict_ = {}
    dict_["Datetime"] = datetime_list
    dict_["Irad"]     = Irad_list
    dict_["Iclr"]     = Iclr_list
    dict_["images"]   = Transformed_images_list
        
    dataframe = pd.DataFrame.from_dict(dict_) 
    return dataframe 



def crop_center(img, cropx=None, cropy=None):
    y, x, _ = img.shape

    if cropy == None:
        cropy = y
    if cropx == None:
        cropx = x

    startx = x//2-(cropx//2)
    starty = y//2-(cropy//2)

    return img[starty:starty+cropy, startx:startx+cropx, :]



def import_and_transform_images(selected_imagepaths_list, image_size=64): 

    train_transforms = transforms.Compose(
    [
        transforms.Resize(image_size, interpolation=transforms.InterpolationMode.BILINEAR), 
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
    ]
    )  

    images = []
    pbar = tqdm(selected_imagepaths_list, total=len(selected_imagepaths_list))
    for image_path in pbar:   

        image_array       = np.asarray(Image.open(image_path))
        cropped_image     = crop_center(image_array, cropx=700, cropy=700)
        transformed_image = train_transforms(Image.fromarray(cropped_image).convert("RGB"))  
        transformed_image_np = transformed_image.numpy()
        images.append(transformed_image_np)
        pbar.set_description("image transformation")

    return images  


def data_preprocessing(irrad_path  = "sirta_data/2023", image_path  = "sirta_data/2023/images", image_size=64):  
        
    image_list               = glob.glob(os.path.join(image_path, "05", "**" , "*_01.jpg"))

    # get image file list
    imagefile_list_of_dict   = {}
    for file_ in image_list:
        imagefile_list_of_dict[os.path.basename(file_)] = file_ 

    imagefile_list           = [os.path.basename(file_) for file_ in image_list]  
    csv_data                 = pd.read_csv(os.path.join(irrad_path, "sirta_data_2023.csv"), index_col=False) 
    all_image_file           = csv_data["Name"].tolist()

    all_image_file_bool      = [] 
    selected_imagenames_list = []
    for file_index, file_ in enumerate(all_image_file): 

        if file_ in imagefile_list:
            all_image_file_bool.append(True) 
            selected_imagenames_list.append(file_)
        else:
            all_image_file_bool.append(False)

    selected_csv_data        = csv_data[all_image_file_bool] 
    selected_imagepaths_list = [imagefile_list_of_dict[imagename_]  for imagename_ in selected_imagenames_list]

    selected_csv_data["Datetime"] = pd.to_datetime(selected_csv_data["Datetime"]) 
    datetime_list                 = selected_csv_data["Datetime"].tolist() 
    Irad_list                     = selected_csv_data["I"].tolist()
    Iclr_list                     = selected_csv_data["Iclr"].tolist() 

    # image reading and processing    
    Transformed_images_list       = import_and_transform_images(selected_imagepaths_list, image_size=image_size) 

    return datetime_list, Irad_list, Iclr_list, Transformed_images_list

 

def mean_square_error(pred_irradiance, target):
    batch_mse = np.sqrt(np.mean((pred_irradiance-target)**2, axis=0))
    return batch_mse[0], batch_mse[-1], np.mean(batch_mse) 



def sample_plot(img, context, img_name):

    plt.figure()
    plt.imshow(img)
    plt.title("[%02d-%02d-%03d-%.2f-%.2f-%.2f]" %(context[0], context[1], context[2], context[3], context[4], context[5]))
    plt.savefig("%s.png" % img_name, bbox_inches='tight')


class AdjustLR:
    def __init__(self, patience = 2, delta=0):
        self.patience = patience 
        self.counter = 0
        self.delta   = delta
        self.best_score = None 
        self.val_loss_min = np.inf 
        self.do_adjust = False
    def __call__(self, val_loss):
        score = -val_loss
        if self.best_score is None:
            self.best_score = score 
            self.do_adjust = False
        elif score < self.best_score + self.delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.do_adjust = True
        else:
            self.best_score = score
            self.do_adjust = False

class EarlyStopping:
    def __init__(self, patience=7, verbose=False, delta=0):
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.inf
        self.delta = delta

    def __call__(self, val_loss, saved_params, configs):
        score = -val_loss
        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, saved_params, configs)
        elif score < self.best_score + self.delta:
            self.counter += 1
            print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, saved_params, configs)
            self.counter = 0

    def save_checkpoint(self, val_loss, saved_params, configs):

        model = saved_params["model"]  

        if self.verbose:
            print(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
        
        model.export_model_for_inferencing(os.path.join(configs["artifacts_dir"], "inference_model.onnx") ,["output"]) 
        self.val_loss_min = val_loss