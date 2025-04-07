import pdb 
import numpy as np
from math import log10, sqrt 
from tqdm import tqdm 
import torch.utils.data as data
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms 
from PIL import Image
from myonnxutils.onnx_utils import list2dataframe, data_preprocessing


class sirta_dataset(data.Dataset): 
    def __init__(self, 
                mode = "Train",
                irrad_path  = "sirta_data/2023",
                image_path  = "sirta_data/2023/images",
                seq_length  = 16,
                pred_length = 15,
                image_size  = 64,
                batch_size  = 64,
                training_index_file = "sirta_data/2023/training.txt",
                validate_index_file = "sirta_data/2023/validate.txt",
                testing_index_file  = "sirta_data/2023/testing.txt"):
        
        super(Dataset, self).__init__() 
         
        self.seq_length          = seq_length
        self.pred_length         = pred_length
        self.image_size          = image_size

        datetime_list, Irad_list, Iclr_list, Transformed_images_list = data_preprocessing(irrad_path  = irrad_path, image_path  = image_path, image_size=image_size)   
        dataframe                                                    = list2dataframe(datetime_list, Irad_list, Iclr_list, Transformed_images_list)     
        input_datetime_stacks, _, input_iclr_stacks, input_image_stacks, output_datetime_stacks, output_irr_stacks, _, _ = self.serialize_data(dataframe)  

        print("Total number of stacked samples: %d" % len(input_datetime_stacks))
 
        if mode == "Train":
            with open(training_index_file, 'r') as f:
                index_list = [line.split("\n")[0] for line in f]
        elif mode == "Valid":
            with open(validate_index_file, 'r') as f:
                index_list = [line.split("\n")[0] for line in f]
        else: 
            with open(testing_index_file, 'r') as f:
                index_list = [line.split("\n")[0] for line in f]

        print("For [%s] mode: the number of stacked samples: %d" % (mode, len(index_list))) 
        print("               the number of batches: %d" % (np.floor(len(index_list)/batch_size))) 
        index_list = [int(index_str) for index_str in index_list] 

        self.input_datetime_stacks      = [input_datetime_stacks[_index_]  for _index_ in index_list]
        self.input_iclr_stacks          = [input_iclr_stacks[_index_]      for _index_ in index_list]
        self.input_image_stacks         = [input_image_stacks[_index_]     for _index_ in index_list]
        self.output_datetime_stacks     = [output_datetime_stacks[_index_] for _index_ in index_list]
        self.output_irr_stacks          = [output_irr_stacks[_index_]      for _index_ in index_list]     
        
    def is_same_date(self, set_datastream):
        flag = True  

        if len(set_datastream["Datetime"].tolist()) < (self.pred_length + self.seq_length):
            flag = False
        else:
            for index, datetime_ in enumerate(set_datastream["Datetime"].tolist()): 
                if index == 0:
                    current_date = datetime_.date()             
                row_date = datetime_.date() 
                
                if not(row_date == current_date):
                    flag = False
        
        return flag
    

    def serialize_data(self, dataframe):
        

        input_datetime_stacks = []
        input_irr_stacks      = [] 
        input_iclr_stacks     = []
        input_image_stacks    = []

        output_datetime_stacks = []
        output_irr_stacks     = []
        output_iclr_stacks    = []
        output_image_stacks   = []


        for index in range(len(dataframe)):
            set_datastream = dataframe.iloc[index : (index + self.seq_length + self.pred_length)]
 
            flag = self.is_same_date(set_datastream)

            if flag == True:

                input_  = dataframe.iloc[index                     : index + self.seq_length]
                output_ = dataframe.iloc[index + self.seq_length   : index + self.seq_length + self.pred_length]
 
                input_datetime_stacks.append(input_["Datetime"].values)
                input_irr_stacks.append(input_["Irad"].values)
                input_iclr_stacks.append(input_["Iclr"].values) 
                input_image_stacks.append(np.stack([input_["images"].values[image_index] for image_index in range(len(input_["images"].values))]).reshape(-1, self.image_size, self.image_size) )
 
                output_datetime_stacks.append(output_["Datetime"].values)
                output_irr_stacks.append(output_["Irad"].values)
                output_iclr_stacks.append(output_["Iclr"].values) 
                output_image_stacks.append(np.stack([output_["images"].values[image_index] for image_index in range(len(output_["images"].values))]).reshape(-1, self.image_size, self.image_size))
                
        return input_datetime_stacks, input_irr_stacks, input_iclr_stacks, input_image_stacks, output_datetime_stacks, output_irr_stacks, output_iclr_stacks, output_image_stacks


    def __len__(self):
        return len(self.input_image_stacks)
    
    def __getitem__(self, idx):
 
        input_iclr     = self.input_iclr_stacks[idx]  
        input_skyimage = self.input_image_stacks[idx]  
        output_irr     = self.output_irr_stacks[idx]   
         
        return idx, input_iclr, input_skyimage, output_irr
    


if __name__ == "__main__":

    from dataloader import sirta_dataset

    dataset = sirta_dataset(    mode = "Train",
                                irrad_path  = "sirta_data/2023",
                                image_path  = "sirta_data/2023/images",
                                seq_length  = 16,
                                pred_length = 15,
                                image_size  = 64,
                                batch_size  = 24,
                                training_index_file = "sirta_data/2023/training.txt",
                                validate_index_file = "sirta_data/2023/validate.txt",
                                testing_index_file  = "sirta_data/2023/testing.txt") 
    
    train_loader = DataLoader(dataset, batch_size=24, shuffle=False, drop_last=True) 

    num_epochs = 5
    for epoch in range(num_epochs):

        progress_bar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{num_epochs}', unit='batch')

        for index, data_batch in enumerate(progress_bar):

            input_index       = data_batch[0]
            input_iclr        = data_batch[1]
            input_skyimagerr  = data_batch[2]
            output_irr        = data_batch[3] 

            pdb.set_trace()