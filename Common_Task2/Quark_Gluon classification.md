# Common Task 2. Quark/Gluon classification:

## Task: To classify input as Quark/Gluon.
--- 

### Dataset:

  [Datasets](https://cernbox.cern.ch/s/oolDBdQegsITFcv)

---
### Approach:

    --> The dataset primarily consisted of files in .test.snappy.parquet format. Each file contained a matrix with a shape of (3,), with each element being an array of 125 elements, and each of those elements containing 125 elements.
    
    --> Due to memory constraints, I wasn't able to load a single file from the 3 files and couldn't use it. So I converted the files into .npy format and split each file into 2 equal halves, then saved them.
    
    --> During the conversion of file formats, I also reshaped elements of X_jets to (3,125,125).
    
    --> Additionally, the dataset included 'pt'(transverse momentum) and 'm0'(invariant mass) values along with the target, which was binary due to the binary classification problem statement.
    
    --> To preprocess the data, I created a DataFrame that encapsulated the matrices of images with shape (3, 125, 125), along with 'pt', 'm0', and the binary label 'y'.
    
    --> The dataset is split into a train and a test set using the `train_test_split()` function from sklearn. The test size is set to 20% of the given data, and the remaining 80% is further divided into 70% for training and 10% for validation.
        

---

### MODELS: → 
#### There are 2 models created: VGG-12, resembling the VGG-16 architecture with a smaller number of layers, and the CustomNet, a custom-designed convolutional neural network featuring three convolutional layers with batch normalization and max-pooling, followed by two fully connected layers for efficient feature extraction and prediction in image-related tasks. 
#
#### Custom_Net():       

##### Drive link for model weights: [Custom_Net()_model_weights](https://drive.google.com/file/d/14U__P3sAqITBH_zUBPh_U7VjcEWgaQyV/view?usp=sharing)

      Custom_Net(
        (conv1): Conv2d(3, 16, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        (bn1): BatchNorm2d(16, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (relu1): ReLU()
        (pool1): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
        (conv2): Conv2d(16, 32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        (bn2): BatchNorm2d(32, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (relu2): ReLU()
        (pool2): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
        (conv3): Conv2d(32, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        (bn3): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (relu3): ReLU()
        (pool3): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
        (fc1): Linear(in_features=14400, out_features=128, bias=True)
        (fc2): Linear(in_features=128, out_features=1, bias=True)
      )

#### VGG-12():

##### Drive link for model weights: [VGG-12()_model_weights](https://drive.google.com/file/d/14Ix1jH_wNfe4DVTp3gS8UFBfVevT29p5/view?usp=drive_link)

      CustomVGG12(
        (features): Sequential(
          (0): Conv2d(3, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
          (1): ReLU(inplace=True)
          (2): Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
          (3): ReLU(inplace=True)
          (4): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
          (5): Conv2d(64, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
          (6): ReLU(inplace=True)
          (7): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
          (8): ReLU(inplace=True)
          (9): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
          (10): Conv2d(128, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
          (11): ReLU(inplace=True)
          (12): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
          (13): ReLU(inplace=True)
          (14): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
        )
        (classifier): Sequential(
          (0): Linear(in_features=57600, out_features=4096, bias=True)
          (1): ReLU(inplace=True)
          (2): Dropout(p=0.5, inplace=False)
          (3): Linear(in_features=4096, out_features=4096, bias=True)
          (4): ReLU(inplace=True)
          (5): Dropout(p=0.5, inplace=False)
          (6): Linear(in_features=4096, out_features=1, bias=True)
        )
      )
    
---

### HyperParameters:
#
#### For CustomNet():

                - Criterion: nn.BCELoss()
                - Optimizer: optim.Adam() with weight decay
                - Number of Epochs: 20
                - Batch Size: 128
                - Learning Rate: 1e-3
                - Weight_decay: 1e-3
                - Scheduler: ReduceLROnPlateau

#### For VGG-12():

                - Criterion: nn.BCEWithLogitsLoss().
                - Optimizer: optim.Adam()
                - Number of Epochs: 10
                - Batch Size: 128
                - Learning Rate: 1e-3
                - Weight_decay: 1e-3
                - Decay_rate: 4e-1
                - Scheduler: ExponentialLR

---

### Results:

#

#### For CustomNet():

                
        Epoch 11/20: 100%|█████████| 230/230 [01:57<00:00, 1.97it/s, training loss=0.5465]
        Val Loss: 0.5874
        Val Acc: 71.30%
        Val ROC-AUC: 0.774
#####     
        100%|██████████| 66/66 [00:01<00:00, 52.71it/s]
        Test Accuracy: 0.7204
        ROC-AUC Score: 0.7804

#### For VGG-12():
                
        Epoch 8/10: 100%|█████████| 230/230 [01:56<00:00,  1.97it/s, train_loss=0.5628] 
        Val Loss: 0.5874
        Val Accuracy: 0.7065
        Val AUC-ROC: 0.7703
#####
        100%|██████████| 66/66 [00:11<00:00,  5.58it/s]
        Test Accuracy: 0.7199
        ROC-AUC Score: 0.7796

#### Model Ensemble:

        100%|██████████| 66/66 [00:12<00:00,  5.34it/s]
        Test Accuracy: 0.7223
        ROC-AUC Score: 0.7807

---

### Below are the Loss, accuracy, and ROC-AUC curves for the architectures, illustrating the point of overfitting and the epoch at which the models were saved.


#
#
| Curves           | Custom_Net()                                                                                                        | VGG-12()                                                                                                        |
|------------------|----------------------------------------------------------------------------------------------------------------------|-----------------------------------------------------------------------------------------------------------------|
| Loss Curve       | <img src="https://github.com/AADI-234/ML4SCI-GSoC2024/assets/133188867/fc523f95-9eba-4c2f-b211-5429edf4c909" width="400" height="330"> | <img src="https://github.com/AADI-234/ML4SCI-GSoC2024/assets/133188867/ec0be8d0-d951-47e1-9eb8-607e429eb799" width="400" height="330"> |
|        |        |
| ROC-AUC Curve    | <img src="https://github.com/AADI-234/ML4SCI-GSoC2024/assets/133188867/92554375-ffee-4d37-9bc4-1aefde950380" width="400" height="330"> | <img src="https://github.com/AADI-234/ML4SCI-GSoC2024/assets/133188867/16d8890f-bc01-4864-a4e2-8473d77ab9ce" width="400" height="330"> |
|        |        |
| Accuracy Curve   | <img src="https://github.com/AADI-234/ML4SCI-GSoC2024/assets/133188867/3318100a-347f-414c-9b40-8e1b2100579e" width="400" height="330"> | <img src="https://github.com/AADI-234/ML4SCI-GSoC2024/assets/133188867/69b33e32-7e1a-4182-99b0-54aac1355263" width="400" height="330"> |

