# <p align="center">ML4SCI_24 </p>


## Common Task 1. Electron/Photon Classification

### Project Resources

| Resource Type          | Description                                       | Link                                                                                        |
|------------------------|---------------------------------------------------|---------------------------------------------------------------------------------------------|
| **Directory**          | Complete collection of project files.             | [Common Task 1](Common_Task1)    |
| **Detailed Solution**  | Approach used     | [Approach](Common_Task1/Electron_photon_classification.md) |
| **Jupyter Notebook**   | Code and analysis in a Jupyter Notebook.      | [Open Notebook](Common_Task1/Common_Task1(cms).ipynb) |
| **PDF Version**        | Pdf of the notebook.                 | [PDF](Common_Task1/Common_Task1(cms).pdf) |
| **Model Weights**      | Model weights for replication and testing.    | [Model_Weights](Common_Task1/model_weights_Common_Task_1.pth)       |

### Results and Analysis

I carefully monitored the training progress over 15 epochs, ensuring optimal performance without overfitting. Below is the conclusion of training:

- **VAL Loss**: 0.2678
- **Val ROC-AUC**: 0.805 
- **Validation Accuracy**: 73.56%
- **Test Loss**: 0.5398
- **Test ROC-AUC**: 0.8044
- **Test Accuracy**: 0.7346


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


## Common Task 2. Quark-Gluon Classification

### Models Overview

| Model Name   | Architecture | Detailed Solution                                                                                     | Notebook                                                                                      | PDF                                                                                     | Model Weights                                                                                                   | Test Accuracy | Test ROC-AUC |
|--------------|--------------|------------------------------------------------------------------------------------------------------|-----------------------------------------------------------------------------------------------|-----------------------------------------------------------------------------------------|-----------------------------------------------------------------------------------------------------------------|---------------|--------------|
| CustomVGG12() | VGG16 based  | [Approach](https://github.com/AADI-234/ML4SCI-GSoC24/blob/main/Common_Task2/Quark_Gluon%20classification.md) | [Notebook](https://github.com/AADI-234/ML4SCI-GSoC24/blob/main/Common_Task2/Common_Task2(cms).ipynb) | [PDF](https://github.com/AADI-234/ML4SCI-GSoC24/blob/main/Common_Task2/Common_Task2(cms).pdf) | [Model Weights](https://drive.google.com/file/d/14Ix1jH_wNfe4DVTp3gS8UFBfVevT29p5/view?usp=sharing) | 71.99%        | 0.7796       |
| Custom_Net() | CNN          | [Approach](https://github.com/AADI-234/ML4SCI-GSoC24/blob/main/Common_Task2/Quark_Gluon%20classification.md) | [Notebook](https://github.com/AADI-234/ML4SCI-GSoC24/blob/main/Common_Task2/Common_Task2(cms).ipynb) | [PDF](https://github.com/AADI-234/ML4SCI-GSoC24/blob/main/Common_Task2/Common_Task2(cms).pdf) | [Model Weights](https://drive.google.com/file/d/14U__P3sAqITBH_zUBPh_U7VjcEWgaQyV/view?usp=sharing) | 72.04%        | 0.7804       |





### Results and Analysis

### Below are the Loss, accuracy, and ROC-AUC curves for the architectures, illustrating the point of overfitting and the epoch at which the models were saved.
#
#
| Curves           | Custom_Net()                                                                                                        | VGG-12()                                                                                                        |
|------------------|----------------------------------------------------------------------------------------------------------------------|-----------------------------------------------------------------------------------------------------------------|
| Loss Curve       | <img src="https://github.com/AADI-234/ML4SCI-GSoC24/assets/133188867/4a69f6e3-4876-4203-8e3a-a45e45d4550e" width="400" height="330"> | <img src="https://github.com/AADI-234/ML4SCI-GSoC24/assets/133188867/1c134977-e9ec-4083-bf30-bd28b59c1346" width="400" height="330"> |
|        |        |
| ROC-AUC Curve    | <img src="https://github.com/AADI-234/ML4SCI-GSoC24/assets/133188867/6f4d5c8c-4c96-40b5-a096-52556aa0e01f" width="400" height="330"> | <img src="https://github.com/AADI-234/ML4SCI-GSoC24/assets/133188867/829a8546-a765-4bf7-9065-d8f0849dba8e" width="400" height="330"> |
|        |        |
| Accuracy Curve   | <img src="https://github.com/AADI-234/ML4SCI-GSoC24/assets/133188867/b64bb4a2-52f8-44fe-a71d-3476e94623c3" width="400" height="330"> | <img src="https://github.com/AADI-234/ML4SCI-GSoC24/assets/133188867/4a633d80-a96c-4aa9-b45f-285867f34563" width="400" height="330"> |


**Observation:** The training curves indicate that careful monitoring of both train and validation losses is crucial to prevent overfitting and to choose the optimal model state for deployment.

------------------------------------------------------------------------------------------------------------------------------------------------------------------
    

	
