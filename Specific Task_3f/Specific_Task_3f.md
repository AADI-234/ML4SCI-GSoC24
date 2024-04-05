# Specific Task 3f:   Event Classification With Masked Transformer Autoencoders 

## Task: To  train a Transformer Autoencoder model to classify and distinguish between a signal process that produces Higgs bosons and a background process that does not.
--- 

### Dataset:

  [Dataset](https://archive.ics.uci.edu/dataset/280/higgs)

---
### Approach:


#### Here's a breakdown of the approach used for the specific task of training a Masked Transformer Autoencoder for classification:


- I used only the first 21 features for training the model as specified.
- The dataset was split into a training set with the first 1.1 million events and a test set with the last 100k items using train_test_split.
- A Transformer Autoencoder model was chosen for this task.
- The encoder utilizes Transformer layers to learn representations of the input data.
- The Encoder output serves as the latent space representation of the input data.
- The Decoder is used for decoding the latent space representation.
- It reconstructs the original input data from the latent space outputs of the encoder.
- A linear layer (bottleneck) is applied to reduce the dimensionality of the latent space representation.
- It compresses the information the encoder learns, potentially improving efficiency and generalization.
- The classifier linear layer predicts the class label based on the bottleneck output.
- It produces a single output node for binary classification tasks, using a sigmoid activation function to generate probabilities.
- The model was trained using the training set with a custom loss function, which is a combination of Mean Squared Error (MSE) loss for autoencoding and Binary Cross Entropy (BCE) loss for classification.
- Metrics such as Validation Loss, Validation Accuracy, and ROC-AUC score were calculated to assess the model's performance.
- The use of learning rate scheduling and gradient clipping helped stabilize training and prevent overfitting.    

---

### MODELS: → 

#### Drive link for best model weights: [Model_Weights](https://drive.google.com/file/d/1KSvzW9kxfD0KmNDo6x9dpvWpWiOi3Rat/view?usp=sharing)
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
                (0-2): 3 x TransformerEncoderLayer(
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
                (0-2): 3 x TransformerDecoderLayer(
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

    - Criterion: nn.MSELoss() + nn.BCELoss                                      
    - Optimizer: optim.AdamW() 
    - Number of Epochs: 17
    - lr = 1e-3
    - batch_size = 1024
    - weight_decay = 5e-4
    - Scheduler:  ReduceLROnPlateau(optimizer, factor=0.4)
    
#### Architecture

    - input_dim = 21
    - latent_dim = 64
    - num_classes = 1
    - num_layers = 3
    - num_heads = 3
    - dropout = 0.1
    
---

### Results:

#### Minimum MAE and MRE Epoch

    Epoch 17/17, Train Loss: 0.2704, Train Acc: 73.86%, Val Loss: 0.2594, Val Acc: 75.26%, Train ROC-AUC: 0.815, Val ROC-AUC: 0.833
    
---

### Below are the Loss curve and the Actual vs predicted output data of the architectures, illustrating the point of overfitting and the epoch at which the models were saved.
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
