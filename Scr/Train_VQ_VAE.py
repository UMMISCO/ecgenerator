import torch.optim as optim
import torch
import time
import numpy as np
from torch import nn


def pearson_correlation(x, y):
    # Ensure that x and y are the same length
    assert len(x) == len(y)

    # Calculate the means of x and y
    x_mean = torch.mean(x, axis = 2)
    y_mean = torch.mean(y, axis = 2)

    # Calculate the variance of x and y
    x_variance = torch.var(x, axis = 2)
    y_variance = torch.var(y, axis = 2)

    # Calculate the standard deviations of x and y
    x_stddev = torch.sqrt(x_variance)
    y_stddev = torch.sqrt(y_variance)



    # Calculate the Pearson correlation coefficient
    r = torch.sum((x - torch.unsqueeze(x_mean, 2)) * (y - torch.unsqueeze(y_mean, 2)), axis = 2 )
    r = r/((len(x[0][0]) * x_stddev * y_stddev))
    r = torch.nan_to_num(r)

    return r

class RMSE_Loss(torch.nn.Module):
    def __init__(self, alpha=1):
        super(RMSE_Loss, self).__init__()
        self.huber_loss = nn.SmoothL1Loss()

    def forward(self, y_true, y_pred):
        #loss1 = torch.mean(torch.square(y_true - y_pred), axis = 2)
        #loss1= torch.nan_to_num(loss1)
        y_true = torch.nan_to_num(y_true)
        y_pred = torch.nan_to_num(y_pred)

        loss2 = pearson_correlation(y_true, y_pred)
        loss2 = torch.nan_to_num(loss2)
        loss2 = torch.clamp_min(loss2, 0)
        
        loss3 = self.huber_loss(y_true, y_pred)
        loss = loss3 + 0.5*(1 - loss2)
        return(torch.mean(torch.mean(loss,axis =1)), torch.mean(loss2), torch.mean(loss3))

def training(epoch, batch_size, train_set,test_set, Verbose = True, device = None, encoder = None, decoder = None, quantizer = None, 
             optimizer_Encoder = None, optimizer_Decoder = None, loss = None, vocab_size = None, skip = 0, size = 3, experiment=0):
    
    lost_list_train = []
    lost_list_test =  []
    corr_list_test = []
    corr_list_train = []
    rmse_list_test = []
    rmse_list_train = []
    for e in range(epoch):
        start_time = time.time()
        train_loss = 0.
        test_loss = 0.
        corr_train = 0.
        corr_test = 0.
        rmse_train = 0.
        rmse_test = 0.
        prec = 0
        it1 = 0
        it2 = 0
        for b in range(0,len(train_set), batch_size):
            data = train_set[b:b + batch_size].astype("float32") 
            data = torch.tensor(data, dtype=torch.float32).to(device)

            
            encoder.zero_grad()
            decoder.zero_grad()

            emb = encoder(data)
            quantized, commit_loss, embed_in, codebook = quantizer(emb.transpose(1,2)) # (1, 1024, 256), (1, 1024), (1)
            
            
            x_hat = decoder(quantized.transpose(1,2))
            lossRMSE, corr, rmse  = loss(data, x_hat)
            lossRMSE = lossRMSE + commit_loss

           


            lossRMSE.backward()
            optimizer_Encoder.step()
            optimizer_Decoder.step()
            
            # update poarameters for Discriminator net
            corr_train += corr.to('cpu').detach().numpy()
            rmse_train += rmse.to('cpu').detach().numpy()
            train_loss += lossRMSE.to('cpu').detach().numpy()
            it1+=1
        lost_list_train.append(train_loss/it1)
        corr_list_train.append(corr_train/it1)
        rmse_list_train.append(rmse_train/it1)
        for b in range(0, len(test_set), batch_size):
            data = test_set[b:b + batch_size].astype("float32") 
            data = torch.tensor(data, dtype=torch.float32).to(device)

            
            encoder.zero_grad()
            decoder.zero_grad()
            encoder.eval()
            decoder.eval()
            with torch.no_grad():
                emb = encoder(data)
                quantized, commit_loss, embed_in, codebook = quantizer(emb.transpose(1,2))
                x_hat = decoder(quantized.transpose(1,2))
                lossRMSE, corr, rmse  = loss(data, x_hat)
            lossRMSE = lossRMSE + commit_loss
            corr_test += corr.to('cpu').detach().numpy()
            rmse_test += rmse.to('cpu').detach().numpy()
            test_loss += lossRMSE.to('cpu').detach().numpy()
            it2+=1
        lost_list_test.append(test_loss/it2)
        corr_list_test.append(corr_test/it2)
        rmse_list_test.append(rmse_test/it2)
        

            
        
        
        if Verbose == True:
            print("Epoch: ", e+1, "/", epoch, " - Time: ", time.time() - start_time, "s")
            print("Train loss: ", train_loss/it1, " - Test loss: ", test_loss/it2)
            print("Train corr: ", corr_train/it1, " - Test corr: ", corr_test/it2)
            print("Train rmse: ", rmse_train/it1, " - Test rmse: ", rmse_test/it2)
        
        
        np.save("../log/VQ-VAE/train_loss_vocab_size_"+str(vocab_size)+"_"+str(size)+"_epoch"+str(e)+".npy", np.array(lost_list_train))
        np.save("../log/VQ-VAE/test_loss_vocab_size_"+str(vocab_size)+"_"+str(size)+"_epoch"+str(e)+".npy", np.array(lost_list_test))
        np.save("../log/VQ-VAE/train_corr_vocab_size_"+str(vocab_size)+"_"+str(size)+"_epoch"+str(e)+".npy", np.array(corr_list_train))
        np.save("../log/VQ-VAE/test_corr_vocab_size_"+str(vocab_size)+"_"+str(size)+"_epoch"+str(e)+".npy", np.array(corr_list_test))
        np.save("../log/VQ-VAE/train_rmse_vocab_size_"+str(vocab_size)+"_"+str(size)+"_epoch"+str(e)+".npy", np.array(rmse_list_train))
        np.save("../log/VQ-VAE/test_rmse_vocab_size_"+str(vocab_size)+"_"+str(size)+"_epoch"+str(e)+".npy", np.array(rmse_list_test))
        
        torch.save(encoder.state_dict(), "../Models/VQ-VAE/Encoder_vocab_size_"+str(vocab_size)+"_"+str(size)+"_epoch"+str(e)+".pth")
        torch.save(decoder.state_dict(), "../Models/VQ-VAE/Decoder_vocab_size_"+str(vocab_size)+"_"+str(size)+"_epoch"+str(e)+".pth")
        torch.save(quantizer.state_dict(), "../Models/VQ-VAE/Quantizer_vocab_size_"+str(vocab_size)+"_"+str(size)+"_epoch"+str(e)+".pth")


    return()