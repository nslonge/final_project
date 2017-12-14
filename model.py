import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.autograd as autograd


class CNN(nn.Module):
	def __init__(self, args, embeddings):
            super(CNN,self).__init__()
            self.args = args
            
            # embedding layer
            V = args.embed_num
            D = args.embed_dim 
            self.embed = nn.Embedding(V, D)
            self.embed.weight.data = torch.from_numpy(embeddings)
            self.embed.weight.requires_grad=False

            # convolution layers
            Ci = 1
            Co = args.kernel_num
            Ks = args.kernel_sizes
            self.convs1 = nn.ModuleList([nn.Conv2d(Ci, Co, (K, D), 
                                                   bias=False) 
                                         for K in Ks])

            # dropout
            self.drop = nn.Dropout2d(0.25)
            
            # convolutional steps for bottleneck regularization
            if 'use_mmd' in self.args.__dict__ and self.args.use_mmd:
                out_size = args.kernel_num * len(args.kernel_sizes)
                b_size = int(args.bottleneck * out_size)
                self.fc1 = nn.Linear(out_size, b_size)
                self.fc2 = nn.Linear(b_size, out_size)

	def forward(self, x):
            x = self.embed(x) 
            x = x.unsqueeze(1)
            # convulational layers 
            x = [F.relu(conv(x)).squeeze(3) for conv in self.convs1] 

            # dropout
            if self.args.dropout: 
                x = [self.drop(i) for i in x]
            
            # pooling
            if self.args.avg_pool:
                x = [F.avg_pool1d(i, i.size(2)).squeeze(2) for i in x] 
            else:
                x = [F.max_pool1d(i, i.size(2)).squeeze(2) for i in x] 
                     
            # concatenate results                               
            x = torch.cat(x, 1)

            # add bottleneck
            if 'use_mmd' in self.args.__dict__ and self.args.use_mmd:
                bottleneck = self.fc1(x)
                x = self.fc2(bottleneck)
                return x, bottleneck
            return x
			

class LSTM(nn.Module):
    def __init__(self, args, embeddings):
        super(LSTM, self).__init__()
        self.args = args
        
        # initialize embedding layer 
        V = args.embed_num
        D = args.embed_dim
        self.embed = nn.Embedding(V, D)
        self.embed.weight.data = torch.from_numpy(embeddings)
        self.embed.weight.requires_grad=False

        # for bidirectional lstm, modify hidden size
        self.hidden_size = args.hidden_size
        if args.bidirectional:
                self.hidden_size = args.hidden_size // 2
        
        self.lstm = nn.LSTM(input_size=D,
                            hidden_size=self.hidden_size,
                            num_layers=args.hidden_layer,
                            batch_first=True,
                            bidirectional=args.bidirectional)
        
        # mmd setup
        if 'use_mmd' in self.args.__dict__ and self.args.use_mmd:
            out_size = args.hidden_size
            b_size = int(args.bottleneck * out_size)
            self.fc1 = nn.Linear(out_size, b_size)
            self.fc2 = nn.Linear(b_size, out_size)

    """
    def sort_batch(self, data, seq_len):
        batch_size = data.size(0)
        sorted_seq_len, sorted_idx = seq_len.sort()
        reverse_idx = torch.linspace(batch_size-1,0,batch_size).long()
        sorted_seq_len = sorted_seq_len[reverse_idx]
        sorted_data = data[sorted_idx][reverse_idx]
        return sorted_data, sorted_seq_len
    """

    def forward(self, x):
        x = self.embed(x) # (N,W,D) 
        if self.args.bidirectional:		# single layer for now
            hidden = (autograd.Variable(torch.zeros(2, 
                                                    len(x), 
                                                    self.hidden_size)),
                      autograd.Variable(torch.zeros(2, 
                                                    len(x), 
                                                    self.hidden_size)))
        else:
            hidden = (autograd.Variable(torch.zeros(self.args.hidden_layer, 
                                                    len(x), 
                                                    self.hidden_size)),
                      autograd.Variable(torch.zeros(self.args.hidden_layer, 
                                                    len(x), 
                                                    self.hidden_size)))
        out, hidden = self.lstm(x, hidden)
        out = out.permute(0,2,1)		
        if self.args.avg_pool:
            out = F.avg_pool1d(out, out.size(2)).squeeze(2)
        else:
            out = F.max_pool1d(out, out.size(2)).squeeze(2)
        if 'use_mmd' in self.args.__dict__ and self.args.use_mmd:
            bottleneck = self.fc1(out)
            out = self.fc2(bottleneck)
                
            return out, bottleneck
        return out


class DomainClassifier(nn.Module):
    def __init__(self, args, embeddings):
        super(DomainClassifier, self).__init__()
        self.args = args
		
        insize = 0
        if args.model == 'lstm':
            insize = args.hidden_size
        else:
            insize = args.kernel_num * len(args.kernel_sizes)

        self.fc1 = nn.Linear(insize, args.domain_size)
        self.fc2 = nn.Linear(args.domain_size, 2)
        self.drop = nn.Dropout2d(0.25)
        self.softmax = nn.LogSoftmax()
    
    def forward(self, x):
        x = self.drop(self.fc1(x))
        x = F.leaky_relu(x)
        x = self.fc2(x)
        x = self.softmax(x)
        return x 
