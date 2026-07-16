# Conditional VAE for Face Expression Generation

An individual implementation of a Conditional Variational Autoencoder (CVAE) built from scratch using TensorFlow and Keras to reconstruct human facial expressions (Happy, Sad, Surprised, and Mad).

> **Note:** The dataset is not included in this repository due to distribution restrictions specified in the assignment.

## Technical Details
* Custom implementation of the Encoder, Decoder, and Sampling layer.
* Custom loss function combining Mean Squared Error and analytical KL Divergence.

## Reconstructions (Original vs. Reconstructed)

### Happy
![Happy Reconstructions](outputs/reconstructions_Happy(1).png)

### Sad
![Sad Reconstructions](outputs/reconstructions_Sad(1).png)

## Quick Start
```bash
conda env create -f environment.yml
conda activate face-expression-cvae

# Run training
python train.py
```
