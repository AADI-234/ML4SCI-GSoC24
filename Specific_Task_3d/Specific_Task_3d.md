# Specific Task 3d:   Masked Auto-Encoder for Efficient E2E Particle Reconstruction and Compression

## Task:

- Train a lightweight ViT using the Masked Auto-Encoder (MAE) training scheme on the unlabelled dataset.
- Compare reconstruction results using MAE on both training and testing datasets.
- Fine-tune the model on a lower learning rate on the provided labelled dataset and compare results with a model trained from scratch.


--- 

### Dataset:

  [Dataset](https://cernbox.cern.ch/s/e3pqxcIznqdYyRv)

---
### Approach:


#### Here's a breakdown of the approach used for the specific task of training a Masked Transformer Autoencoder for classification:


- Trained a lightweight Vision Transformer (ViT) using a Masked Auto-Encoder (MAE) training scheme on the unlabelled dataset.
- The model architecture includes Transformer layers, self-attention mechanisms, masking, and custom loss function.
- Trained using AdamW optimizer with hyperparameter tuning, layer normalization, gradient clipping, and dropout regularization.
- Utilized custom loss function combining masking and Mean Squared Error Loss (MSELoss) for autoencoder reconstruction.
- Fine-tuned on the labeled dataset with a lower learning rate to adapt to task-specific information.
- Evaluated performance using Reconstruction Loss on both training and testing datasets.
- Compared results between fine-tuning and training from scratch.
- Conducted linear probing with and without pretraining.
- Monitored metrics on both training and validation datasets to prevent overfitting and assess generalization.
- Saved the best model based on minimum Reconstruction Loss for future use

---

### MODELS: → 
#### The Masked_VIT model consists of an encoder with patch embedding and Transformer blocks, followed by a decoder with linear embedding, Transformer blocks, and a prediction layer.
#
    Masked_VIT(
      (encoder): Encoder(
        (patch_embed): PatchEmbed(
          (proj): Conv2d(8, 768, kernel_size=(5, 5), stride=(5, 5))
          (norm): Identity()
        )
        (blocks): ModuleList(
          (0-7): 8 x Block(
            (norm1): LayerNorm((768,), eps=1e-06, elementwise_affine=True)
            (attn): Attention(
              (qkv): Linear(in_features=768, out_features=2304, bias=True)
              (q_norm): Identity()
              (k_norm): Identity()
              (attn_drop): Dropout(p=0.0, inplace=False)
              (proj): Linear(in_features=768, out_features=768, bias=True)
              (proj_drop): Dropout(p=0.0, inplace=False)
            )
            (ls1): Identity()
            (drop_path1): Identity()
            (norm2): LayerNorm((768,), eps=1e-06, elementwise_affine=True)
            (mlp): Mlp(
              (fc1): Linear(in_features=768, out_features=3072, bias=True)
              (act): GELU(approximate='none')
              (drop1): Dropout(p=0.0, inplace=False)
              (norm): Identity()
              (fc2): Linear(in_features=3072, out_features=768, bias=True)
              (drop2): Dropout(p=0.0, inplace=False)
            )
            (ls2): Identity()
            (drop_path2): Identity()
          )
        )
        (norm): LayerNorm((768,), eps=1e-06, elementwise_affine=True)
      )
      (decoder): Decoder(
        (decoder_embed): Linear(in_features=768, out_features=512, bias=True)
        (decoder_blocks): ModuleList(
          (0-3): 4 x Block(
            (norm1): LayerNorm((512,), eps=1e-06, elementwise_affine=True)
            (attn): Attention(
              (qkv): Linear(in_features=512, out_features=1536, bias=True)
              (q_norm): Identity()
              (k_norm): Identity()
              (attn_drop): Dropout(p=0.0, inplace=False)
              (proj): Linear(in_features=512, out_features=512, bias=True)
              (proj_drop): Dropout(p=0.0, inplace=False)
            )
            (ls1): Identity()
            (drop_path1): Identity()
            (norm2): LayerNorm((512,), eps=1e-06, elementwise_affine=True)
            (mlp): Mlp(
              (fc1): Linear(in_features=512, out_features=2048, bias=True)
              (act): GELU(approximate='none')
              (drop1): Dropout(p=0.0, inplace=False)
              (norm): Identity()
              (fc2): Linear(in_features=2048, out_features=512, bias=True)
              (drop2): Dropout(p=0.0, inplace=False)
            )
            (ls2): Identity()
            (drop_path2): Identity()
          )
        )
        (decoder_norm): LayerNorm((512,), eps=1e-06, elementwise_affine=True)
        (decoder_pred): Linear(in_features=512, out_features=200, bias=True)
      )
    )

---

### HyperParameters:
#
#### MAE Model hyperparameters

    - Criterion: nn.MSELoss()                                    
    - Optimizer: optim.AdamW() 
    - Number of Epochs: 80
    - lr = 1.5e-4
    - batch_size = 64
    - weight_decay = 5e-2
    - Scheduler:  CosineAnnealingWarmRestarts
    
#### MAE Architecture

    - input_dim 8
    - latent_dim: 768
    - num_classes: 2
    - num_layers:
      - Encoder: 8
      - Decoder: 4
    - num_heads: 1 (for both encoder and decoder)

#### Both trained and pre-trained models are fine-tuned on a learning rate of 1.e-5 using AdamW optimizer.

---

### Results:

#### Fine tuned Model with pre-training

    100%|██████████| 32/32 [00:26<00:00,  1.20it/s]
    Epoch 8/8, Train Loss: 0.2604, Train Accuracy: 0.8494, Valid Loss: 0.3750, Valid Accuracy: 0.8548

#### Fine tuned Model without pre-training 

    
    100%|██████████| 32/32 [00:26<00:00,  1.21it/s]
    Epoch 15/15, Train Loss: 0.5238, Train Accuracy: 0.6813, Valid Loss: 0.5265, Valid Accuracy: 0.6951
        
---

### Below are the Loss curve and the Actual vs predicted output data of the architectures, illustrating the point of overfitting and the epoch at which the models were saved.

| Curves           | Model with Pre-training                                                                                                        | Model without Pre-training                                                                                                       |
|------------------|----------------------------------------------------------------------------------------------------------------------|-----------------------------------------------------------------------------------------------------------------|
| Loss Curve       | <img src="https://github.com/AADI-234/ML4SCI-GSoC24/assets/133188867/2034e189-44bf-4e30-b514-1ba0d0d330ad" width="400" height="330"> | <img src="https://github.com/AADI-234/ML4SCI-GSoC24/assets/133188867/eb00e9c8-11c7-4728-bdfe-cd8bba792128" width="400" height="330"> |

- Monitors the model's convergence during training. A decreasing loss indicates learning progress, while sudden increases may indicate overfitting.
