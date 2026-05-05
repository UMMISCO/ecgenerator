import torch.optim as optim
import torch
import time
import numpy as np
from torch import nn
from torch.utils.data import DataLoader, TensorDataset


def accuracy_fn(logits, targets):
    return (logits == targets).float().mean().item()

def training(model, encoder, quantizer, train_data, test_data, optimizer, device, vocab_size, profondeur,
             epochs, batch_size=32, temperature=1.0, top_k=None):
    
    model.to(device)
    model.train()
    
    list_loss_train = []
    list_acc_train = []

    list_loss_test = []
    list_acc_test = []

    dataset = TensorDataset(torch.tensor(train_data.astype(np.float32)))
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    dataset_test = TensorDataset(torch.tensor(test_data.astype(np.float32)))
    dataloader_test = DataLoader(dataset_test, batch_size=batch_size, shuffle=False)

    encoder.to(device)
    quantizer.to(device)
    
    quantizer.eval()
    encoder.eval()



    for epoch in range(epochs):
        total_loss_train = 0
        total_acc_train = 0
        for batch in dataloader:
            batch_input = batch[0].to(device)
            out = encoder(batch_input)
            out = out.transpose(1,2)
            out,_,true_ind,_ = quantizer(out)

            rd = np.random.randint(1, true_ind.shape[1] - 1)
            inputs = true_ind[:, :rd]
            targets = true_ind[:, 1:rd + 1]
  
            logits, loss = model(inputs, targets=targets, device = device)

            B,T = targets.shape
            targets = targets.reshape(B*T)
            logits = torch.argmax(logits, axis=-1)
            acc = accuracy_fn(logits, targets)

            # Backward
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss_train += loss.item()
            total_acc_train += acc
        avg_loss_train = total_loss_train / len(dataloader)
        avg_acc_train = total_acc_train / len(dataloader)
        list_loss_train.append(avg_loss_train)
        list_acc_train.append(avg_acc_train)

        total_loss_test = 0
        total_acc_test = 0
        for batch in dataloader_test:
            batch_input = batch[0].to(device)
            out = encoder(batch_input)
            out = out.transpose(1,2)
            out,_,true_ind,_ = quantizer(out)

            rd = np.random.randint(1, true_ind.shape[1] - 1)
            inputs = true_ind[:, :rd]
            targets = true_ind[:, 1:rd + 1]

            logits, loss = model(inputs, targets=targets, device = device)

            B,T = targets.shape
            targets = targets.reshape(B*T)
            logits = torch.argmax(logits, axis=-1)
            acc = accuracy_fn(logits, targets)

            total_loss_test += loss.item()
            total_acc_test += acc
        avg_loss_test = total_loss_test / len(dataloader_test)
        avg_acc_test = total_acc_test / len(dataloader_test)
        list_loss_test.append(avg_loss_test)
        list_acc_test.append(avg_acc_test)


        if epoch % 10 == 0 or epoch == epochs - 1:
            print(f"Epoch {epoch + 1}/{epochs}, Train Loss: {avg_loss_train:.4f}, Train Acc: {avg_acc_train:.4f}, Test Loss: {avg_loss_test:.4f}, Test Acc: {avg_acc_test:.4f}")
            torch.save(model.state_dict(),"../Models/NanoGPT/Transformer_"+str(vocab_size)+"_"+str(profondeur)+".pth")
            np.save("../log/NanoGPT/loss_test_"+str(vocab_size)+"_"+str(profondeur)+".npy", np.array(list_loss_test))
            np.save("../log/NanoGPT/acc_test_"+str(vocab_size)+"_"+str(profondeur)+".npy", np.array(list_acc_test))
            np.save("../log/NanoGPT/loss_train_"+str(vocab_size)+"_"+str(profondeur)+".npy", np.array(list_loss_train))
            np.save("../log/NanoGPT/acc_train_"+str(vocab_size)+"_"+str(profondeur)+".npy", np.array(list_acc_train))

        torch.save(model.state_dict(),"../Models/NanoGPT/Transformer_"+str(vocab_size)+"_"+str(profondeur)+".pth")
        np.save("../log/NanoGPT/loss_test_"+str(vocab_size)+"_"+str(profondeur)+".npy", np.array(list_loss_test))
        np.save("../log/NanoGPT/acc_test_"+str(vocab_size)+"_"+str(profondeur)+".npy", np.array(list_acc_test))
        np.save("../log/NanoGPT/loss_train_"+str(vocab_size)+"_"+str(profondeur)+".npy", np.array(list_loss_train))
        np.save("../log/NanoGPT/acc_train_"+str(vocab_size)+"_"+str(profondeur)+".npy", np.array(list_acc_train))