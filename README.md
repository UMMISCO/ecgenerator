# ECGenerator

This README explains how to run the training process for the **ECGenerator** model.

## Project Structure

The `src` folder contains all the Python scripts required to train the models.

## Training Order

It is important to follow this order:

1. Train the **VQ-VAE** model
2. Train the **Transformer (NanoGPT)** model

## VQ-VAE Training

To start training the VQ-VAE model, run the notebook:

```
Main_VQ_VAE.ipynb
```

Default parameters:

* Number of epochs: 30
* Batch size: 128

Data is loaded using the `Load_Data.py` script.

During training, the weights of the encoder, decoder, and quantizer are saved in:

```
Models/VQ-VAE
```

## Transformer (NanoGPT) Training

Once the VQ-VAE is trained, you can train the Transformer by running:

```
Main_Transformer
```

⚠️ Important: you must provide the pre-trained weights of the encoder and quantizer.

The Transformer model weights will be saved in:

```
Models/NanoGPT
```
