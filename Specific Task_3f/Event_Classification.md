# Specific Task 3a Regression  (Using DeepViT)

## Task: To  train a Transformer Autoencoder model to classify and distinguish between a signal process that produces Higgs bosons and a background process that does not.
--- 

### Dataset:

  [Dataset](https://archive.ics.uci.edu/dataset/280/higgs)

---
### Approach:
    

---

### MODELS: → 

#### Drive link for best model weights based on min. Val Loss: [Model_Weights](https://drive.google.com/file/d/1brB-RCGIdFRt2MjIJsljlrDtOJdcU8OK/view?usp=sharing)
#### Drive link for best model weights based on min. MRE error: [Model_Weights](https://drive.google.com/file/d/1Mm1plLo4SOha8tOEjpF8kCDYw68OGAJj/view?usp=sharing)
#
#### The model is a Transformer-based autoencoder with an added classifier, designed for feature extraction and classification tasks on input data with 21 features, where the bottleneck layer reduces the dimensionality to 64 before classification.
#
    TransformerAutoencoderClassifier(
      (encoder_layers): TransformerEncoderLayer(
        (self_attn): MultiheadAttention(
          (out_proj): NonDynamicallyQuantizableLinear(in_features=21, out_features=21, bias=True)
        )
        (linear1): Linear(in_features=21, out_features=2048, bias=True)
        (dropout): Dropout(p=0.1, inplace=False)
        (linear2): Linear(in_features=2048, out_features=21, bias=True)
        (norm1): LayerNorm((21,), eps=1e-05, elementwise_affine=True)
        (norm2): LayerNorm((21,), eps=1e-05, elementwise_affine=True)
        (dropout1): Dropout(p=0.1, inplace=False)
        (dropout2): Dropout(p=0.1, inplace=False)
      )
      (encoder): TransformerEncoder(
        (layers): ModuleList(
          (0-1): 2 x TransformerEncoderLayer(
            (self_attn): MultiheadAttention(
              (out_proj): NonDynamicallyQuantizableLinear(in_features=21, out_features=21, bias=True)
            )
            (linear1): Linear(in_features=21, out_features=2048, bias=True)
            (dropout): Dropout(p=0.1, inplace=False)
            (linear2): Linear(in_features=2048, out_features=21, bias=True)
            (norm1): LayerNorm((21,), eps=1e-05, elementwise_affine=True)
            (norm2): LayerNorm((21,), eps=1e-05, elementwise_affine=True)
            (dropout1): Dropout(p=0.1, inplace=False)
            (dropout2): Dropout(p=0.1, inplace=False)
          )
        )
      )
      (decoder_layers): TransformerDecoderLayer(
        (self_attn): MultiheadAttention(
          (out_proj): NonDynamicallyQuantizableLinear(in_features=21, out_features=21, bias=True)
        )
        (multihead_attn): MultiheadAttention(
          (out_proj): NonDynamicallyQuantizableLinear(in_features=21, out_features=21, bias=True)
        )
        (linear1): Linear(in_features=21, out_features=2048, bias=True)
        (dropout): Dropout(p=0.1, inplace=False)
        (linear2): Linear(in_features=2048, out_features=21, bias=True)
        (norm1): LayerNorm((21,), eps=1e-05, elementwise_affine=True)
        (norm2): LayerNorm((21,), eps=1e-05, elementwise_affine=True)
        (norm3): LayerNorm((21,), eps=1e-05, elementwise_affine=True)
        (dropout1): Dropout(p=0.1, inplace=False)
        (dropout2): Dropout(p=0.1, inplace=False)
        (dropout3): Dropout(p=0.1, inplace=False)
      )
      (decoder): TransformerDecoder(
        (layers): ModuleList(
          (0-1): 2 x TransformerDecoderLayer(
            (self_attn): MultiheadAttention(
              (out_proj): NonDynamicallyQuantizableLinear(in_features=21, out_features=21, bias=True)
            )
            (multihead_attn): MultiheadAttention(
              (out_proj): NonDynamicallyQuantizableLinear(in_features=21, out_features=21, bias=True)
            )
            (linear1): Linear(in_features=21, out_features=2048, bias=True)
            (dropout): Dropout(p=0.1, inplace=False)
            (linear2): Linear(in_features=2048, out_features=21, bias=True)
            (norm1): LayerNorm((21,), eps=1e-05, elementwise_affine=True)
            (norm2): LayerNorm((21,), eps=1e-05, elementwise_affine=True)
            (norm3): LayerNorm((21,), eps=1e-05, elementwise_affine=True)
            (dropout1): Dropout(p=0.1, inplace=False)
            (dropout2): Dropout(p=0.1, inplace=False)
            (dropout3): Dropout(p=0.1, inplace=False)
          )
        )
      )
      (bottleneck): Linear(in_features=21, out_features=64, bias=True)
      (classifier): Linear(in_features=64, out_features=1, bias=True)
      (dropout): Dropout(p=0.1, inplace=False)
    )
---

### HyperParameters:
#
#### Model

    Criterion: nn.MSELoss() + nn.BCELoss                                      
    Optimizer: optim.AdamW() 
    Number of Epochs: 15
    lr = 1e-3
    batch_size = 1024
    weight_decay = 5e-4
    Scheduler:  ReduceLROnPlateau(optimizer, factor=0.4)
    
#### Architecture


    input_dim = 21
    latent_dim = 64
    num_classes = 1
    num_layers = 2
    num_heads = 3
    dropout = 0.1
    
---

### Results:

#### Minimum MAE and MRE Epoch

        Epoch 15/15, Train Loss: 0.2724, Train Acc: 73.70%, Val Loss: 0.2617, Val Acc: 74.99%, Train ROC-AUC: 0.813, Val ROC-AUC: 0.830
---

### Below are the Loss curve and the Actual vs predicted output data of the architectures, illustrating the point of overfitting and the epoch at which the models were saved.
#
## Loss Curve
![Loss Curve](https://github.com/AADI-234/ML4SCI-GSoC24/assets/133188867/2da5628d-2dc3-4427-b725-d4f075a124b2)
- Monitors the model's convergence during training. A decreasing loss indicates learning progress, while sudden increases may indicate overfitting.


## ROC-AUC Curve
![ROC-AUC Curve](https://github.com/AADI-234/ML4SCI-GSoC24/assets/133188867/5a7e9469-fb04-4e48-9c35-7652feab3d77)
- Evaluates the model's ability to distinguish between positive and negative classes in binary classification tasks. Higher AUC scores indicate better discrimination performance.


## Accuracy Curve

![Accuracy Curve](https://github.com/AADI-234/ML4SCI-GSoC24/assets/133188867/e54cd0b5-820b-46ff-badd-334f185b3be7)
- Tracks the model's performance on the training and validation datasets. Helps assess how well the model generalizes to unseen data.