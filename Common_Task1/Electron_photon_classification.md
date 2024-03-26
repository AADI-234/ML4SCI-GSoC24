# Common Task 1. Electron/photon classification:

## Task: To classify input as Electron/Photons.
--- 

### Dataset:

  [Photons Dataset](https://cernbox.cern.ch/index.php/s/AtBT8y4MiQYFcgc)  
  [Electrons Dataset](https://cernbox.cern.ch/index.php/s/FbXw3V4XNyYB3oA)

---

### Approach:


    --> The dataset mainly consisted of files in .hdf5 format. Each data file contained matrices with a shape of 32x32x2, where the 2 represented the number of channels. These channels represented hit energy and time.
    
    --> Firstly, I reshaped the matrix into 2x32x32 so that it could be fed into the neural network.
      
    --> Created a `DatasetLoader` class that returns the data loaders.
    
    --> The dataset is split into a train and a test set using the `train_test_split()` function from sklearn. The test size is set to 20% of the given data, and the remaining 80% is further divided into 70% for training and 10% for validation.
    
---

### MODEL: → Replicated ResNet architecture with 15 layers forming the ResNet15.

#### Drive link for model weights: [ResNet15()_model_weights](https://drive.google.com/file/d/1A1yROMo3UKt2JocxmVsfNa7-cVCRGykp/view?usp=drive_link)
    
    ResNet15(
      (conv1): Conv2d(2, 16, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
      (bn1): BatchNorm2d(16, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (layer1): Sequential(
        (0): BasicBlock(
          (conv1): Conv2d(16, 16, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
          (bn1): BatchNorm2d(16, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          (conv2): Conv2d(16, 16, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
          (bn2): BatchNorm2d(16, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          (shortcut): Sequential()
        )
        (1): BasicBlock(
          (conv1): Conv2d(16, 16, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
          (bn1): BatchNorm2d(16, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          (conv2): Conv2d(16, 16, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
          (bn2): BatchNorm2d(16, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          (shortcut): Sequential()
        )
      )
      (layer2): Sequential(
        (0): BasicBlock(
          (conv1): Conv2d(16, 32, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False)
          (bn1): BatchNorm2d(32, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          (conv2): Conv2d(32, 32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
          (bn2): BatchNorm2d(32, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          (shortcut): Sequential(
            (0): Conv2d(16, 32, kernel_size=(1, 1), stride=(2, 2), bias=False)
            (1): BatchNorm2d(32, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          )
        )
        (1): BasicBlock(
          (conv1): Conv2d(32, 32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
          (bn1): BatchNorm2d(32, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          (conv2): Conv2d(32, 32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
          (bn2): BatchNorm2d(32, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          (shortcut): Sequential()
        )
      )
      (layer3): Sequential(
        (0): BasicBlock(
          (conv1): Conv2d(32, 64, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False)
          (bn1): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          (conv2): Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
          (bn2): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          (shortcut): Sequential(
            (0): Conv2d(32, 64, kernel_size=(1, 1), stride=(2, 2), bias=False)
            (1): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          )
        )
        (1): BasicBlock(
          (conv1): Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
          (bn1): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          (conv2): Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
          (bn2): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          (shortcut): Sequential()
        )
      )
      (linear): Linear(in_features=64, out_features=1, bias=True)
    )


---

### HyperParameters:

     - Criterion: nn.BCELoss()
     - Optimizer: optim.Adam()
     - Number of Epochs: 15
     - Batch Size: 32
     - Learning Rate: 1e-3
     - Scheduler: CosineAnnealingWarmRestarts
---

### Results:

      Epoch 15/15 100%|█████████| 12450/12450 [02:19<00:00, 89.57it/s, training loss=0.5249]
      VAL Loss 0.2678
      Val ROC-AUC: 0.805 
      Validation Accuracy: 73.56%
#
      Test Loss: 0.5398
      Test ROC-AUC: 0.8044
      Test Accuracy: 0.7346

      

---

#### Below are the Loss, accuracy, and ROC-AUC curves for the architectures, illustrating the point of overfitting and the epoch at which the models were saved.

#

![Loss Curve](https://github.com/AADI-234/ML4SCI-GSoC24/assets/133188867/6fc8ed40-465b-4858-9ca7-c58cecf521c1)
- Monitors the model's convergence during training. A decreasing loss indicates learning progress, while sudden increases may indicate overfitting.


#

![ROC-AUC Curve](https://github.com/AADI-234/ML4SCI-GSoC24/assets/133188867/95c0aae7-6928-4ca5-ade3-4ea9d595f3c0)
- Evaluates the model's ability to distinguish between positive and negative classes in binary classification tasks. Higher AUC scores indicate better discrimination performance.


# 

![Accuracy Curve](https://github.com/AADI-234/ML4SCI-GSoC24/assets/133188867/ab9c590a-2134-4797-8f44-3fd6a5942e14)
- Tracks the model's performance on the training and validation datasets. Helps assess how well the model generalizes to unseen data.




