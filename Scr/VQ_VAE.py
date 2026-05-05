import numpy as np
from torch import nn
from torch.nn import functional as F
import torch


class Encoder_net(nn.Module):
    def __init__(self,nbr_block):
        super(Encoder_net, self).__init__()
        
        chanel_out = 64
        chanel_in = 1
        kernel_size = 8
        stride = 2
        padding = 0
        self.__block = []
        for i in range(nbr_block):
            layer = nn.Sequential(
                nn.Conv1d(chanel_in, chanel_out,kernel_size = kernel_size, stride = stride, padding = padding),
                nn.BatchNorm1d(chanel_out),
                #nn.Dropout(0.3),
                nn.Conv1d(chanel_out, chanel_out,kernel_size = 3, stride = 1, padding = 1),
                nn.BatchNorm1d(chanel_out),
                #nn.Dropout(0.3),
                nn.ELU(0.01)
            )
            self.__block.append(layer)
            self.add_module(f"ConvBlock_{i}", layer)
            chanel_in = chanel_out
            if chanel_out >= 2048:
                chanel_out = 2048
            else:
                chanel_out = int(chanel_out*2)
            
            
    def forward(self, inp):
        out = inp
        for layer in self.__block:
            out = layer(out)
        return(out)


class Decoder_net(nn.Module):
    def __init__(self,nbr_block, chanel_in = 512):
        super(Decoder_net, self).__init__()
        
        chanel_in = chanel_in
        chanel_out = 512
        kernel_size = 8
        stride = 2
        padding = 0
        self.__block = []
        for i in range(nbr_block):
            layer = nn.Sequential(
                nn.ConvTranspose1d(chanel_in, chanel_out,kernel_size = kernel_size, stride = stride, padding = padding),
                #nn.BatchNorm1d(chanel_out),
                nn.ELU(0.01)
            )
            self.add_module(f"DeconvBlock_{i}", layer)
            self.__block.append(layer)
            chanel_in = chanel_out
            chanel_out = int(chanel_out/2)
            
        self.last_layer = nn.ConvTranspose1d(chanel_in, 1,kernel_size = 22, stride = stride, padding = padding)
        self.last_activation = nn.Tanh()

            
    def forward(self, inp):
        out = inp
        for layer in self.__block:
            out = layer(out)
        out = self.last_layer(out)
        out = self.last_activation(out)

        if out.shape[-2] != 5000:
            out = F.interpolate(out, size=5000, mode='linear', align_corners=True)
        return(out)


class Quantize_net(nn.Module):
    def __init__(self, dim, n_embed, experiment = 42, decay=0.99, eps=1e-5):
        super().__init__()

        self.dim = dim
        self.n_embed = n_embed # taille du dictionnaire
        self.decay = decay
        self.eps = eps
        torch.manual_seed(experiment)
        embed = torch.randn(dim, n_embed)
        self.register_buffer("embed", embed)
        self.register_buffer("cluster_size", torch.zeros(n_embed))
        self.register_buffer("embed_avg", embed.clone())

    def forward(self, inp):
        flatten = inp.reshape(-1, self.dim)
        dist = (
            flatten.pow(2).sum(1, keepdim=True)
            - 2 * flatten @ self.embed
            + self.embed.pow(2).sum(0, keepdim=True)
        )
        _, embed_ind = (-dist).max(1)
        embed_onehot = F.one_hot(embed_ind, self.n_embed).type(flatten.dtype)
        embed_ind = embed_ind.view(*inp.shape[:-1])
        quantize = self.embed_code(embed_ind)

        if self.training:
            embed_onehot_sum = embed_onehot.sum(0)
            embed_sum = flatten.transpose(0, 1) @ embed_onehot

            #dist_fn.all_reduce(embed_onehot_sum)
            #dist_fn.all_reduce(embed_sum)

            self.cluster_size.data.mul_(self.decay).add_(
                embed_onehot_sum, alpha=1 - self.decay
            )
            self.embed_avg.data.mul_(self.decay).add_(embed_sum, alpha=1 - self.decay)
            n = self.cluster_size.sum()
            cluster_size = (
                (self.cluster_size + self.eps) / (n + self.n_embed * self.eps) * n
            )
            embed_normalized = self.embed_avg / cluster_size.unsqueeze(0)
            self.embed.data.copy_(embed_normalized)

        diff = (quantize.detach() - inp).pow(2).mean()
        quantize = inp + (quantize - inp).detach()
        
        return quantize, diff, embed_ind, self.embed
    
    def embed_code(self, embed_id):
        return F.embedding(embed_id, self.embed.transpose(0, 1))


