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
- **Test Accuracy**: 73.46%


#### Below are the Loss, accuracy, and ROC-AUC curves for the architectures, illustrating the point of overfitting and the epoch at which the models were saved.

#

![Loss Curve](https://github.com/AADI-234/ML4SCI-GSoC24/assets/133188867/6fc8ed40-465b-4858-9ca7-c58cecf521c1)
- Monitors the model's convergence during training. A decreasing loss indicates learning progress, while sudden increases may indicate overfitting.


#

![ROC-AUC Curve](https://github.com/AADI-234/ML4SCI-GSoC24/assets/133188867/95c0aae7-6928-4ca5-ade3-4ea9d595f3c0)
- Evaluates the model's ability to distinguish between positive and negative classes in binary classification tasks. Higher AUC scores indicate better discrimination performance.


# 

![Accuracy Curve](https://github.com/AADI-234/ML4SCI-GSoC24/assets/133188867/ab9c590a-2134-4797-8f44-3fd6a5942e14) 
- Tracks the model's performance on the training, validation and test datasets. Helps assess how well the model generalizes to unseen data.

---
--- 

## Common Task 2. Quark-Gluon Classification

### Models Overview

| Model Name   | Architecture | Detailed Solution                                                                                     | Notebook                                                                                      | PDF                                                                                     | Model Weights                                                                                                   | Test Accuracy | Test ROC-AUC |
|--------------|--------------|------------------------------------------------------------------------------------------------------|-----------------------------------------------------------------------------------------------|-----------------------------------------------------------------------------------------|-----------------------------------------------------------------------------------------------------------------|---------------|--------------|
| CustomVGG12() | VGG16 based  | [Approach](https://github.com/AADI-234/ML4SCI-GSoC24/blob/main/Common_Task2/Quark_Gluon%20classification.md) | [Notebook](https://github.com/AADI-234/ML4SCI-GSoC24/blob/main/Common_Task2/Common_Task2(cms).ipynb) | [PDF](https://github.com/AADI-234/ML4SCI-GSoC24/blob/main/Common_Task2/Common_Task2(cms).pdf) | [Model Weights](https://drive.google.com/file/d/14Ix1jH_wNfe4DVTp3gS8UFBfVevT29p5/view?usp=sharing) | 71.99%        | 0.7796       |
| Custom_Net() | CNN          | [Approach](https://github.com/AADI-234/ML4SCI-GSoC24/blob/main/Common_Task2/Quark_Gluon%20classification.md) | [Notebook](https://github.com/AADI-234/ML4SCI-GSoC24/blob/main/Common_Task2/Common_Task2(cms).ipynb) | [PDF](https://github.com/AADI-234/ML4SCI-GSoC24/blob/main/Common_Task2/Common_Task2(cms).pdf) | [Model Weights](https://drive.google.com/file/d/14U__P3sAqITBH_zUBPh_U7VjcEWgaQyV/view?usp=sharing) | 72.04%        | 0.7804       |
| Model_Ensemble() | Ensemble | [Approach](https://github.com/AADI-234/ML4SCI-GSoC24/blob/main/Common_Task2/Quark_Gluon%20classification.md) | [Notebook](https://github.com/AADI-234/ML4SCI-GSoC24/blob/main/Common_Task2/Common_Task2(cms).ipynb) | [PDF](https://github.com/AADI-234/ML4SCI-GSoC24/blob/main/Common_Task2/Common_Task2(cms).pdf) |       - | 72.23%        | 0.7807       |

### Results and Analysis

### Below are the Loss, accuracy, and ROC-AUC curves for the architectures, illustrating the point of overfitting and the epoch at which the models were saved.
#
#
| Curves           | Custom_Net()                                                                                                        | VGG-12()                                                                                                        |
|------------------|----------------------------------------------------------------------------------------------------------------------|-----------------------------------------------------------------------------------------------------------------|
| Loss Curve       | <img src="https://github.com/AADI-234/ML4SCI-GSoC24/assets/133188867/4a69f6e3-4876-4203-8e3a-a45e45d4550e" width="400" height="330"> | <img src="https://github.com/AADI-234/ML4SCI-GSoC24/assets/133188867/1c134977-e9ec-4083-bf30-bd28b59c1346" width="400" height="330"> |
|        |        |
| Accuracy Curve   | <img src="https://github.com/AADI-234/ML4SCI-GSoC24/assets/133188867/6f4d5c8c-4c96-40b5-a096-52556aa0e01f" width="400" height="330"> | <img src="https://github.com/AADI-234/ML4SCI-GSoC24/assets/133188867/829a8546-a765-4bf7-9065-d8f0849dba8e" width="400" height="330"> |
|        |        |
| ROC-AUC Curve    | <img src="https://github.com/AADI-234/ML4SCI-GSoC24/assets/133188867/b64bb4a2-52f8-44fe-a71d-3476e94623c3" width="400" height="330"> | <img src="https://github.com/AADI-234/ML4SCI-GSoC24/assets/133188867/4a633d80-a96c-4aa9-b45f-285867f34563" width="400" height="330"> |


**Observation:** The training curves indicate that careful monitoring of both train and validation losses is crucial to prevent overfitting and to choose the optimal model state for deployment.

---
---

## Specific Task 3a. Deep Learning Regression

### Models Overview

| Model Name   | Architecture | Detailed Solution                                                                                     | Notebook                                                                                      | PDF                                                                                     | Model Weights                                                                                                   | Val Loss | MRE |
|--------------|--------------|------------------------------------------------------------------------------------------------------|-----------------------------------------------------------------------------------------------|-----------------------------------------------------------------------------------------|-----------------------------------------------------------------------------------------------------------------|---------------|--------------|
| CustomResNet18() | Resnet  | [Approach](Specific_Task_3a/CNN_Regression.md) | [Notebook](https://github.com/AADI-234/ML4SCI-GSoC24/blob/main/Specific_Task_3a/Task_3a%20%20(using%20CNN)%20.ipynb) | [PDF](https://github.com/AADI-234/ML4SCI-GSoC24/blob/main/Specific_Task_3a/Task_3a%20%20(using%20CNN)%20.pdf) | [Model Weights](https://drive.google.com/file/d/10VIbOaqa_gFXPt5DDRdzcrZMqhK32Azq/view) | 0.6502        | 2.08964       |
| DeepViT() | Vision Transformer          | [Approach](Specific_Task_3a/DeepViT_Regression.md) | [Notebook](https://github.com/AADI-234/ML4SCI-GSoC24/blob/main/Specific_Task_3a/Task_3a%20%20(using%20DeepViT)%20%20.ipynb) | [PDF](https://github.com/AADI-234/ML4SCI-GSoC24/blob/main/Specific_Task_3a/Task_3a%20%20(using%20DeepViT)%20%20.pdf) | [Model Weights](https://drive.google.com/file/d/1brB-RCGIdFRt2MjIJsljlrDtOJdcU8OK/view) | 1.0094        | 39.713       |

#
### Results and Analysis

I carefully monitored the training progress over 25 epochs both the models, ensuring optimal performance without overfitting. Below is the conclusion of training :

#### Using CNN
	Minimum MAE and MRE Epoch
	
	        Epoch 13/25 (Training): 100%|██████████| 96/96 [00:21<00:00,  4.49it/s]
	        Epoch 13/25 (Validation): 100%|██████████| 24/24 [00:01<00:00, 17.20it/s]
	        Epoch 13/25, Train Loss: 0.3180, Val Loss: 0.6502, MAE: 0.6505, MRE: 2.0864
	 
	Minimum Train Loss Epoch
 
		Epoch 25/25 (Training): 100%|██████████| 96/96 [00:21<00:00,  4.51it/s]
		Epoch 25/25 (Validation): 100%|██████████| 24/24 [00:01<00:00, 17.30it/s]
		Epoch 25/25, Train Loss: 0.0335, Val Loss: 0.7684, MAE: 0.6859, MRE: 2.5986
 #
#### Using ViT
	Minimum MAE and MRE Epoch
	
	        Epoch 3/25 (Training): 100%|██████████| 192/192 [00:17<00:00, 10.93it/s]
	        Epoch 3/25 (Validation): 100%|██████████| 48/48 [00:01<00:00, 28.29it/s]
	        Epoch 3/25, Train Loss: 0.9926, Val Loss: 1.0094, MAE: 27.8555, MRE: 39.7913
	
	Minimum Train Loss Epoch
	        
	        Epoch 23/25 (Training): 100%|██████████| 192/192 [00:17<00:00, 10.86it/s]
	        Epoch 23/25 (Validation): 100%|██████████| 48/48 [00:01<00:00, 28.19it/s]
	        Epoch 23/25, Train Loss: 0.6072, Val Loss: 1.1940, MAE: 29.4635, MRE: 70.7605
#
### Below are the Loss curve and the Actual vs predicted output data of the architectures, illustrating the point of overfitting and the epoch at which the models were saved.


|  Predictions         | ResNet18                                                                                                        | DeepViT                                                                                                       |
|------------------|----------------------------------------------------------------------------------------------------------------------|-----------------------------------------------------------------------------------------------------------------|
| Loss Curve | <img src="https://github.com/AADI-234/ML4SCI-GSoC24/assets/133188867/4a11f3bf-9912-43a9-b775-8204be13a17a" width="400" height="330"> | <img src="https://github.com/AADI-234/ML4SCI-GSoC24/assets/133188867/142377ad-b36b-4a77-ae6b-69da25eeeff0)" width="400" height="330"> |
|        |        |
| On Training data   | <img src="https://github.com/AADI-234/ML4SCI-GSoC24/assets/133188867/be509a42-1fb7-4f0c-8fef-b7c8b622c595" width="400" height="330"> | <img src="https://github.com/AADI-234/ML4SCI-GSoC24/assets/133188867/f55c9c72-3d07-4b09-a822-4ac65184d998" width="400" height="330"> |
|        |        |
| On Validation data   | <img src="https://github.com/AADI-234/ML4SCI-GSoC24/assets/133188867/880b4ee6-b031-4c7c-b704-178545a59f51" width="400" height="330"> | <img src="https://github.com/AADI-234/ML4SCI-GSoC24/assets/133188867/9940b188-e71f-421e-9220-7f3ab41d60b9" width="400" height="330"> |
|        |        |

---

## Specific Task 3f: 

### Project Resources

| Resource Type          | Description                                       | Link                                                                                        |
|------------------------|---------------------------------------------------|---------------------------------------------------------------------------------------------|
| **Directory**          | Complete collection of project files.             | [Specific_Task 3f](https://github.com/AADI-234/ML4SCI-GSoC24/tree/main/Specific%20Task_3f)    |
| **Detailed Solution**  | Approach used     | [Approach](https://github.com/AADI-234/ML4SCI-GSoC24/blob/main/Specific%20Task_3f/Event_Classification.md) |
| **Jupyter Notebook**   | Code and analysis in a Jupyter Notebook.      | [Open Notebook](https://github.com/AADI-234/ML4SCI-GSoC24/blob/main/Specific%20Task_3f/Specific_Task_3f.ipynb) |
| **PDF Version**        | Pdf of the notebook.                 | [PDF](https://github.com/AADI-234/ML4SCI-GSoC24/blob/main/Specific%20Task_3f/Specific_Task_3f.pdf) |
| **Model Weights**      | Model weights for replication and testing.    | [Model_Weights](https://github.com/AADI-234/ML4SCI-GSoC24/blob/main/Specific%20Task_3f/model_weights_Transformer_Autoencoder%20.pth)       |

### Results and Analysis

I carefully monitored the training progress over 15 epochs, ensuring optimal performance without overfitting. Below is the conclusion of training:

- **VAL Loss**: 0.2594
- **Val ROC-AUC**: 0.833
- **Validation Accuracy**: 75.26%
- **Train Loss**: 0.2704
- **Train ROC-AUC**: 0.815
- **Train Accuracy**: 73.86%


#### Below are the Loss, accuracy, and ROC-AUC curves for the architectures, illustrating the point of overfitting and the epoch at which the models were saved.

#
## Loss Curve
![Loss Curve](https://github.com/AADI-234/ML4SCI-GSoC24/assets/133188867/41c1c33f-c5d7-43e4-9fce-ef64e9dcc760)
- Monitors the model's convergence during training. A decreasing loss indicates learning progress, while sudden increases may indicate overfitting.


## ROC-AUC Curve
![ROC-AUC Curve](https://github.com/AADI-234/ML4SCI-GSoC24/assets/133188867/3751290d-5816-469f-9acf-5e1bc49f584f)
- Evaluates the model's ability to distinguish between positive and negative classes in binary classification tasks. Higher AUC scores indicate better discrimination performance.

## Accuracy Curve

![Accuracy Curve](https://github.com/AADI-234/ML4SCI-GSoC24/assets/133188867/afec5856-c601-46a0-a920-7e08a7c27496)
- Tracks the model's performance on the training and validation datasets. Helps assess how well the model generalizes to unseen data.

---

## Specific Task 3d: Masked Auto-Encoder for Efficient End-to-End Particle Reconstruction and Compression

### Project Resources


### Tasks
1. Train a lightweight ViT using the Masked Auto-Encoder (MAE) training scheme on the unlabelled dataset.
2. Compare reconstruction results using MAE on both training and testing datasets.
3. Fine-tune the model on a lower learning rate on the provided labelled dataset and compare results with a model trained from scratch.

<p align="center">
  <img src="https://github.com/Wodlfvllf/End-to-End-Deep-Learning-Project/blob/main/Specific%20Task%203d%20-%20Masked_Autoencoder/MAE.png" width="700" title="hover text">
</p>

### Implementation
- Trained a lightweight ViT using MAE on unlabelled dataset
- Compared reconstruction results on training and testing datasets
- Fine-tuned the model on a lower learning rate using the labelled dataset
- Compared results with a model trained from scratch
- Ensured no overfitting on the test dataset

### Image Reconstruction
####                                           Original
<p align="center">
  <img src="https://github.com/Wodlfvllf/End-to-End-Deep-Learning-Project/blob/main/Specific%20Task%203d%20-%20Masked_Autoencoder/Original.jpg" width="700" title="hover text">
</p>

####                                           Reconstructed
<p align="center">
  <img src="https://github.com/Wodlfvllf/End-to-End-Deep-Learning-Project/blob/main/Specific%20Task%203d%20-%20Masked_Autoencoder/Reconstructed.jpg" width="700" title="hover text">
</p>

### Comparison of With and Without Pretrained Vision Transformer Model
                          | Model               | Accuracy |
                          |---------------------|----------|
                          | With Pretrained     | 0.8548   |
                          | Without Pretrained  | 0.7151   |
                          
Both models are fine-tuned on learning rate of 1.e-5 using AdamW optimizer.

- [MAE_Particle_Reconstruction.ipynb](https://github.com/Wodlfvllf/End-to-End-Deep-Learning-Project/blob/main/Specific%20Task%203d%20-%20Masked_Autoencoder/Masked_Autoencoder/Masked%20Autoencoder.ipynb)
- [linear-probing-Pretraining.ipynb](https://github.com/Wodlfvllf/End-to-End-Deep-Learning-Project/blob/main/Specific%20Task%203d%20-%20Masked_Autoencoder/Linear%20Probing%20MAE/linear-probing-Pretraining.ipynb)
- [linear-probing-without Pretraining.ipynb](https://github.com/Wodlfvllf/End-to-End-Deep-Learning-Project/blob/main/Specific%20Task%203d%20-%20Masked_Autoencoder/Linear%20Probing%20MAE/linear-probing-without%20Pretraining.ipynb)
- Includes data loading, model training (pre-training and fine-tuning), evaluation, and model weights

------------------------------------------------------------------------------------------------------------------------------------------------------------------
    

	
