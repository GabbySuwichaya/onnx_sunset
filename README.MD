
## Onnx inference and training

We have exported a pretrained sunset model into Onnx format.

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


