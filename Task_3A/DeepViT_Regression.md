# Specific Task 3a Regression

## Task: To  train a model to estimate (regress) the mass of the particle based on particle images using the provided dataset. 
--- 

### Dataset:

  [Dataset](https://cernbox.cern.ch/s/zUvpkKhXIp0MJ0g)

---
### Approach:
    
- **To effectively handle large datasets within memory constraints, I employed a strategy of chunking with a chunk size of 8.** This approach maximizes data utilization while addressing memory limitations. Additionally, I sorted the data according to the criteria outlined in the research paper ([link](https://arxiv.org/abs/2204.12313)), where the conditions were defined as follows:
    
- **Condition:** \( pT,A = 20–100 \) GeV, \( mA = 0–1.6 \) GeV, and \( |\eta A| < 1.4 \)[here](https://arxiv.org/abs/2204.12313)

- The dataset predominantly comprised files in the `.test.snappy.parquet` format. Each file contained a matrix with dimensions of (8,), where each element was an array of 125 elements, and each of those elements contained 125 sub-elements.

- During the conversion of file formats, I reshaped the elements of X_jets to (8, 125, 125).

- **For modeling, instead of using a regular Vision Transformer (ViT), I opted for a DeepViT architecture, as described in the research paper ([link](https://arxiv.org/abs/2103.11886)).** DeepViT addresses the saturation issue encountered in ViTs by mixing the attention of each head post-softmax, providing better performance without requiring an excessive increase in depth.

- In the DeepViT model, the first four channels of X_jets were used for the output prediction.

- Subsequently, I split the dataset into training and testing sets using the `train_test_split()` function from the sklearn library. The test size was designated as 20% of the total data, leaving the remaining 80% for training purposes.

- Finally, I trained the DeepViT model on GPUs for faster training, leveraging their computational power to expedite the training process and achieve better performance.


---

### MODELS: → 

#### Drive link for best model weights based on min. Val Loss: [Model_Weights](https://drive.google.com/file/d/1brB-RCGIdFRt2MjIJsljlrDtOJdcU8OK/view?usp=sharing)
#### Drive link for best model weights based on min. MRE error: [Model_Weights](https://drive.google.com/file/d/1Mm1plLo4SOha8tOEjpF8kCDYw68OGAJj/view?usp=sharing)
#
#### The DeepViT model is a variant of Vision Transformer (ViT) tailored for mass regression. It includes multiple layers of deep transformers, each consisting of multi-head self-attention mechanisms and feedforward neural networks. The model is designed to handle 4-channel input images and predict a single scalar mass value.
#
    DeepViT(
      (to_patch_embedding): Sequential(
        (0): Rearrange('b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1=25, p2=25)
        (1): LayerNorm((2500,), eps=1e-05, elementwise_affine=True)
        (2): Linear(in_features=2500, out_features=1024, bias=True)
        (3): LayerNorm((1024,), eps=1e-05, elementwise_affine=True)
      )
      (dropout): Dropout(p=0.1, inplace=False)
      (transformer): Transformer(
        (layers): ModuleList(
          (0): ModuleList(
            (0): Attention(
              (norm): LayerNorm((1024,), eps=1e-05, elementwise_affine=True)
              (to_qkv): Linear(in_features=1024, out_features=1536, bias=False)
              (dropout): Dropout(p=0.1, inplace=False)
              (reattn_norm): Sequential(
                (0): Rearrange('b h i j -> b i j h')
                (1): LayerNorm((8,), eps=1e-05, elementwise_affine=True)
                (2): Rearrange('b i j h -> b h i j')
              )
              (to_out): Sequential(
                (0): Linear(in_features=512, out_features=1024, bias=True)
                (1): Dropout(p=0.1, inplace=False)
              )
            )
            (1): FeedForward(
              (net): Sequential(
                (0): LayerNorm((1024,), eps=1e-05, elementwise_affine=True)
                (1): Linear(in_features=1024, out_features=2048, bias=True)
                (2): GELU(approximate='none')
                (3): Dropout(p=0.1, inplace=False)
                (4): Linear(in_features=2048, out_features=1024, bias=True)
                (5): Dropout(p=0.1, inplace=False)
              )
            )
          )
          (1): ModuleList(
            (0): Attention(
              (norm): LayerNorm((1024,), eps=1e-05, elementwise_affine=True)
              (to_qkv): Linear(in_features=1024, out_features=1536, bias=False)
              (dropout): Dropout(p=0.1, inplace=False)
              (reattn_norm): Sequential(
                (0): Rearrange('b h i j -> b i j h')
                (1): LayerNorm((8,), eps=1e-05, elementwise_affine=True)
                (2): Rearrange('b i j h -> b h i j')
              )
              (to_out): Sequential(
                (0): Linear(in_features=512, out_features=1024, bias=True)
                (1): Dropout(p=0.1, inplace=False)
              )
            )
            (1): FeedForward(
              (net): Sequential(
                (0): LayerNorm((1024,), eps=1e-05, elementwise_affine=True)
                (1): Linear(in_features=1024, out_features=2048, bias=True)
                (2): GELU(approximate='none')
                (3): Dropout(p=0.1, inplace=False)
                (4): Linear(in_features=2048, out_features=1024, bias=True)
                (5): Dropout(p=0.1, inplace=False)
              )
            )
          )
          (2): ModuleList(
            (0): Attention(
              (norm): LayerNorm((1024,), eps=1e-05, elementwise_affine=True)
              (to_qkv): Linear(in_features=1024, out_features=1536, bias=False)
              (dropout): Dropout(p=0.1, inplace=False)
              (reattn_norm): Sequential(
                (0): Rearrange('b h i j -> b i j h')
                (1): LayerNorm((8,), eps=1e-05, elementwise_affine=True)
                (2): Rearrange('b i j h -> b h i j')
              )
              (to_out): Sequential(
                (0): Linear(in_features=512, out_features=1024, bias=True)
                (1): Dropout(p=0.1, inplace=False)
              )
            )
            (1): FeedForward(
              (net): Sequential(
                (0): LayerNorm((1024,), eps=1e-05, elementwise_affine=True)
                (1): Linear(in_features=1024, out_features=2048, bias=True)
                (2): GELU(approximate='none')
                (3): Dropout(p=0.1, inplace=False)
                (4): Linear(in_features=2048, out_features=1024, bias=True)
                (5): Dropout(p=0.1, inplace=False)
              )
            )
          )
          (3): ModuleList(
            (0): Attention(
              (norm): LayerNorm((1024,), eps=1e-05, elementwise_affine=True)
              (to_qkv): Linear(in_features=1024, out_features=1536, bias=False)
              (dropout): Dropout(p=0.1, inplace=False)
              (reattn_norm): Sequential(
                (0): Rearrange('b h i j -> b i j h')
                (1): LayerNorm((8,), eps=1e-05, elementwise_affine=True)
                (2): Rearrange('b i j h -> b h i j')
              )
              (to_out): Sequential(
                (0): Linear(in_features=512, out_features=1024, bias=True)
                (1): Dropout(p=0.1, inplace=False)
              )
            )
            (1): FeedForward(
              (net): Sequential(
                (0): LayerNorm((1024,), eps=1e-05, elementwise_affine=True)
                (1): Linear(in_features=1024, out_features=2048, bias=True)
                (2): GELU(approximate='none')
                (3): Dropout(p=0.1, inplace=False)
                (4): Linear(in_features=2048, out_features=1024, bias=True)
                (5): Dropout(p=0.1, inplace=False)
              )
            )
          )
          (4): ModuleList(
            (0): Attention(
              (norm): LayerNorm((1024,), eps=1e-05, elementwise_affine=True)
              (to_qkv): Linear(in_features=1024, out_features=1536, bias=False)
              (dropout): Dropout(p=0.1, inplace=False)
              (reattn_norm): Sequential(
                (0): Rearrange('b h i j -> b i j h')
                (1): LayerNorm((8,), eps=1e-05, elementwise_affine=True)
                (2): Rearrange('b i j h -> b h i j')
              )
              (to_out): Sequential(
                (0): Linear(in_features=512, out_features=1024, bias=True)
                (1): Dropout(p=0.1, inplace=False)
              )
            )
            (1): FeedForward(
              (net): Sequential(
                (0): LayerNorm((1024,), eps=1e-05, elementwise_affine=True)
                (1): Linear(in_features=1024, out_features=2048, bias=True)
                (2): GELU(approximate='none')
                (3): Dropout(p=0.1, inplace=False)
                (4): Linear(in_features=2048, out_features=1024, bias=True)
                (5): Dropout(p=0.1, inplace=False)
              )
            )
          )
          (5): ModuleList(
            (0): Attention(
              (norm): LayerNorm((1024,), eps=1e-05, elementwise_affine=True)
              (to_qkv): Linear(in_features=1024, out_features=1536, bias=False)
              (dropout): Dropout(p=0.1, inplace=False)
              (reattn_norm): Sequential(
                (0): Rearrange('b h i j -> b i j h')
                (1): LayerNorm((8,), eps=1e-05, elementwise_affine=True)
                (2): Rearrange('b i j h -> b h i j')
              )
              (to_out): Sequential(
                (0): Linear(in_features=512, out_features=1024, bias=True)
                (1): Dropout(p=0.1, inplace=False)
              )
            )
            (1): FeedForward(
              (net): Sequential(
                (0): LayerNorm((1024,), eps=1e-05, elementwise_affine=True)
                (1): Linear(in_features=1024, out_features=2048, bias=True)
                (2): GELU(approximate='none')
                (3): Dropout(p=0.1, inplace=False)
                (4): Linear(in_features=2048, out_features=1024, bias=True)
                (5): Dropout(p=0.1, inplace=False)
              )
            )
          )
        )
      )
      (to_latent): Identity()
      (mlp_head): Sequential(
        (0): LayerNorm((1024,), eps=1e-05, elementwise_affine=True)
        (1): Linear(in_features=1024, out_features=1, bias=True)
      )
    )
---

### HyperParameters:
#
#### Model
    - Criterion: nn.MSELoss()                                            
    - Optimizer: optim.Adam() 
    - Number of Epochs: 25
    - Batch Size: 32
    - Learning Rate: 1e-3
    - weight_decay: 1e-4
    - Scheduler: ExponentialLR(optimizer, gamma=0.4)
    
#### Architecture

    - patch_size = 25,
    - depth = 6,
    - heads = 8,
    - dropout = 0.1,
    - emb_dropout = 0.1
    - dim = 1024,
    - mlp_dim = 2048,
    

---

### Results:

#### Minimum MAE and MRE Epoch

        Epoch 3/25 (Training): 100%|██████████| 192/192 [00:17<00:00, 10.93it/s]
        Epoch 3/25 (Validation): 100%|██████████| 48/48 [00:01<00:00, 28.29it/s]
        Epoch 3/25, Train Loss: 0.9926, Val Loss: 1.0094, MAE: 27.8555, MRE: 39.7913

#### Minimum Train Loss Epoch
        
        Epoch 23/25 (Training): 100%|██████████| 192/192 [00:17<00:00, 10.86it/s]
        Epoch 23/25 (Validation): 100%|██████████| 48/48 [00:01<00:00, 28.19it/s]
        Epoch 23/25, Train Loss: 0.6072, Val Loss: 1.1940, MAE: 29.4635, MRE: 70.7605

---

### Below are the Loss curve and the Actual vs predicted output data of the architectures, illustrating the point of overfitting and the epoch at which the models were saved.
#
## Loss Curve
![Loss Curve](https://github.com/AADI-234/ML4SCI-GSoC24/assets/133188867/6f7eadeb-6cb7-4bb9-b25e-31768a11dd03)
- Monitors the model's convergence during training. A decreasing loss indicates learning progress, while sudden increases may indicate overfitting.

## Predictions (On Training Data)
#### Based on min. Val Loss
![image](https://github.com/AADI-234/ML4SCI-GSoC24/assets/133188867/be3b6f97-979a-41c8-a0e0-9c51ffab3eb7)
#### Based on min. MRE error
![image](https://github.com/AADI-234/ML4SCI-GSoC24/assets/133188867/cdd6add5-0db5-417f-af9a-fbba730c15e7)


## Predictions (On Validation Data)   
![image](https://github.com/AADI-234/ML4SCI-GSoC24/assets/133188867/6a7c5204-93f6-43ea-9aec-1c730f713532)


