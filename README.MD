
## Onnx inference and training

We have exported a pretrained sunset model into Onnx format.

- You can download the exported model from [sunset_onnex_trained_torch](https://drive.google.com/drive/folders/111ocu9zzFG1kZ7sLcCnYcgb9W_PFA0kA?usp=sharing)

- An incompleted set of SIRTA data is used and can be downloaded from [INCOMPLETE_SIRTA_DATA](https://drive.google.com/drive/folders/1q6KAiL3TYvcZWi-sMRHkbdozudbFlpYC?usp=sharing)
 

- After you have run all the files, you should get the following directories:

    ```
    - configs/
        - sunset_configs.yaml
    - myonnxutils/
        - onnx_utils.py
    - sirta_data/
        - 2023/
            - images/...
            - sirta_data_2023.csv
    - sunset_onnx/
        - artifacts/
            - checkpoint
            - eval_model.onnx
            - inference_model.onnx
            - optimizer_model.onnx
            - training_model.onnx
    - sunset_onnx_trained_torch/
        - param_names.txt
        - sunset_model.onnx 
    ```

### Pre-requisite
 

1. The following packages are required for performing inference (during deployment):

    ```
    conda create -n onnx_sunset pip install python==3.11
    pip3 install torch torchvision torchaudio
    pip3 install onnxruntime-gpu==1.19 
    python -m pip install -U matplotlib
    pip install netron
    ```

2. (Optional) if you want to train the Onnx model on GPU, you will need to install: 
    - CUDA 11.x
    - CUDNN 8.x 
    - Then, install the following package for training:
    ```
    pip3 install onnxruntime-training
    ```

### Inferring with an onnx model

This repo will give you the instruction on how to:

- Perform the inference             `step_1_inference_onnx_sunset.ipynb`
- Perform the fine-tunning on Onnx  `step_2_train_artifact_sunset.ipynb`


