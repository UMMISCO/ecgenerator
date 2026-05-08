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

## ECG Generation

Once all models are trained, you can generate synthetic ECG signals using the full VQ-VAE + GPT pipeline.

### Setup

Initialize and load all models with their pretrained weights:

```python
vocab_size = 64
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Initialize models
encoder   = Encoder_net(4)
decoder   = Decoder_net(3, chanel_in=encoding_size)
quantizer = Quantize_net(encoding_size, vocab_size)
NanoGPT   = GPTLanguageModel(block_size=306, vocab_size=vocab_size,
                              n_embd=384, n_layer=10, n_head=10,
                              dropout=0.2, num_classes=12, device=device)

# Load pretrained weights
encoder   = load_model(encoder,   "Models/VQ-VAE/Encoder_vocab_size_64_4_epoch0.pth",   device)
decoder   = load_model(decoder,   "Models/VQ-VAE/Decoder_vocab_size_64_4_epoch0.pth",   device)
quantizer = load_model(quantizer, "Models/VQ-VAE/Quantizer_vocab_size_64_4_epoch0.pth", device)
NanoGPT   = load_model(NanoGPT,   "Models/NanoGPT/Transformer_64_10.pth",               device)
```

### Generate ECGs

Use a few real ECG samples as seed context to generate new synthetic signals:

```python
generated_ecgs = generate_ecg(
    encoder, decoder, quantizer, NanoGPT,
    nb_ecg=10,          # Number of generation iterations
    device=device,
    ecg=torch.tensor(subset[:2]).to(device),  # Seed ECGs used as context
    block_size=77       # Number of tokens used as GPT context
)
```

### Generation Pipeline

Internally, `generate_ecg` follows these steps for each iteration:

1. **Encode** — the seed ECG is passed through the VQ-VAE encoder
2. **Quantize** — the encoded representation is mapped to discrete token indices via the codebook
3. **Generate** — NanoGPT autoregressively generates a new token sequence (up to 306 tokens) from the first `block_size` context tokens
4. **Decode** — the generated token indices are mapped back to continuous embeddings using the codebook
5. **Reconstruct** — the VQ-VAE decoder reconstructs the final ECG signal from the embeddings

### Output

`generate_ecgs` is a list of tensors, each of shape `(batch, channels, length)`, on CPU.

You can visualize a generated ECG as follows:

```python
plt.plot(generated_ecgs[0][0].tolist())
plt.title("Generated ECG")
plt.xlabel("Time")
plt.ylabel("Amplitude")
plt.show()
```

> ⚠️ **Note:** `.tolist()` must be used instead of `.numpy()` when running on Apple Silicon (MPS backend), as NumPy interoperability is not available in that environment.
