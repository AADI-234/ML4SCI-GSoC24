# Specific Task 3a Regression

## Task: To  train a model to estimate (regress) the mass of the particle based on particle images using the provided dataset. 

---

### Dataset:

  [Dataset](https://cernbox.cern.ch/s/zUvpkKhXIp0MJ0g)

---
### Approach:

- **To effectively handle large datasets within memory constraints, I employed a strategy of chunking with a chunk size of 8.** This approach maximizes data utilization while addressing memory limitations. Additionally, I sorted the data according to the criteria outlined in the research paper ([link](https://arxiv.org/abs/2204.12313)), where the conditions were defined as follows:

- **Condition:** \( pT,A = 20–100 \) GeV, \( mA = 0–1.6 \) GeV, and \( |\eta A| < 1.4 \) ([link](https://arxiv.org/abs/2204.12313)),

- The dataset predominantly comprised files in the `.test.snappy.parquet` format. Each file contained a matrix with dimensions of `(8,)`, where each element was an array of 125 elements, and each of those elements contained 125 sub-elements.

- During the conversion of file formats, I reshaped the elements of `X_jets` to `(8,125,125)`.

- I utilized the first four channels of `X_jets` for the output prediction.

- Subsequently, I split the dataset into training and testing sets using the `train_test_split()` function from the sklearn library. The test size was designated as 20% of the total data, leaving the remaining 80% for training purposes.

- Finally, I trained the Custom_ResNet18 model on parallel GPUs (GPU T4*2) for faster training, leveraging their computational power to expedite the training process and achieve better performance. 


---

### MODELS: → 

#### Drive link for best model weights based on min. Val Loss: [Model_Weights](https://drive.google.com/file/d/1Y4a1OTwlBX7vekJZkUj2QuiIx3ciLYO4/view?usp=sharing)
#### Drive link for best model weights based on min. MRE error: [Model_Weights](https://drive.google.com/file/d/10VIbOaqa_gFXPt5DDRdzcrZMqhK32Azq/view?usp=sharing)
#

#### The CustomResNet18 model is a variant of ResNet18 tailored for mass regression. It includes standard convolutional and residual blocks, followed by a sequence of fully connected layers for regression, predicting a single scalar mass value from a 4-channel input image.
#
    CustomResNet18(
      (conv1): Conv2d(4, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
      (bn1): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (relu): ReLU(inplace=True)
      (maxpool): MaxPool2d(kernel_size=3, stride=2, padding=1, dilation=1, ceil_mode=False)
      (layer1): Sequential(
        (0): BasicBlock(
          (conv1): Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
          (bn1): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          (conv2): Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
          (bn2): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          (shortcut): Sequential()
        )
        (1): BasicBlock(
          (conv1): Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
          (bn1): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          (conv2): Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
          (bn2): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          (shortcut): Sequential()
        )
      )
      (layer2): Sequential(
        (0): BasicBlock(
          (conv1): Conv2d(64, 128, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False)
          (bn1): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          (conv2): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
          (bn2): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          (shortcut): Sequential(
            (0): Conv2d(64, 128, kernel_size=(1, 1), stride=(2, 2), bias=False)
            (1): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          )
        )
        (1): BasicBlock(
          (conv1): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
          (bn1): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          (conv2): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
          (bn2): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          (shortcut): Sequential()
        )
      )
      (layer3): Sequential(
        (0): BasicBlock(
          (conv1): Conv2d(128, 256, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False)
          (bn1): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          (conv2): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
          (bn2): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          (shortcut): Sequential(
            (0): Conv2d(128, 256, kernel_size=(1, 1), stride=(2, 2), bias=False)
            (1): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          )
        )
        (1): BasicBlock(
          (conv1): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
          (bn1): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          (conv2): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
          (bn2): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          (shortcut): Sequential()
        )
      )
      (fc1): Linear(in_features=65536, out_features=512, bias=True)
      (fc2): Linear(in_features=512, out_features=256, bias=True)
      (fc3): Linear(in_features=256, out_features=128, bias=True)
      (fc4): Linear(in_features=128, out_features=1, bias=True)
    )
---

### HyperParameters:
#

    - Criterion: nn.MSELoss()
    - Optimizer: optim.Adam() 
    - Number of Epochs: 25
    - Batch Size: 64
    - Learning Rate: 5e-4
    - Scheduler: CosineAnnealingLR

---

### Results:

#
        Epoch 13/25 (Training): 100%|██████████| 96/96 [00:21<00:00,  4.49it/s]
        Epoch 13/25 (Validation): 100%|██████████| 24/24 [00:01<00:00, 17.20it/s]
        Epoch 13/25, Train Loss: 0.3180, Val Loss: 0.6502, MAE: 0.6505, MRE: 2.0864

#
        Epoch 25/25 (Training): 100%|██████████| 96/96 [00:21<00:00,  4.51it/s]
        Epoch 25/25 (Validation): 100%|██████████| 24/24 [00:01<00:00, 17.30it/s]
        Epoch 25/25, Train Loss: 0.0335, Val Loss: 0.7684, MAE: 0.6859, MRE: 2.5986

---

### Below are the Loss curve and the Actual vs predicted output data of the architectures, illustrating the point of overfitting and the epoch at which the models were saved.



## Loss Curve
![Loss Curve](https://github.com/AADI-234/ML4SCI-GSoC24/assets/133188867/4a11f3bf-9912-43a9-b775-8204be13a17a)
- Monitors the model's convergence during training. A decreasing loss indicates learning progress, while sudden increases in Validation Loss indicate overfitting.
#
|  Predictions         | Min. ValLoss Model                                                                                                        | Min. MRE Model                                                                                                       |
|------------------|----------------------------------------------------------------------------------------------------------------------|-----------------------------------------------------------------------------------------------------------------|
| On Training data       | <img src="https://github.com/AADI-234/ML4SCI-GSoC24/assets/133188867/2734c25a-382a-467e-abcb-ff9701a5ab94" width="400" height="330"> | <img src="https://github.com/AADI-234/ML4SCI-GSoC24/assets/133188867/be509a42-1fb7-4f0c-8fef-b7c8b622c595" width="400" height="330"> |
|        |        |
| On Validation data   | <img src="https://github.com/AADI-234/ML4SCI-GSoC24/assets/133188867/adf194ea-11f5-46ea-8f25-48ee6636b3a7" width="400" height="330"> | <img src="https://github.com/AADI-234/ML4SCI-GSoC24/assets/133188867/880b4ee6-b031-4c7c-b704-178545a59f51" width="400" height="330"> |
|        |        |
