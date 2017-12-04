import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.autograd as autograd
import pdb

class CNN(nn.Module):
	def __init__(self, args, embeddings):
            super(CNN,self).__init__()
            self.args = args
            
            V = args.embed_num
            D = args.embed_dim
            Ci = 1
            Co = args.kernel_num
            Ks = args.kernel_sizes

            # initialize embeddingl layer 
            self.embed = nn.Embedding(V, D)
            self.embed.weight.data = torch.from_numpy(embeddings)
            self.embed.weight.requires_grad=False

            # convolution step
            self.convs1 = nn.ModuleList([nn.Conv2d(Ci, Co, (K, D), bias=False) for K in Ks])

	def forward(self, x):
		
            x = self.embed(x) # (N,W,D) (bs, 38, 200)

            x = x.unsqueeze(1) # (N,Ci,W,D) (bs, 1, 38, 200)

            x = [F.relu(conv(x)).squeeze(3) for conv in self.convs1] 
                                     #[(N,Co,W), ...]*len(Ks) (bs, 100, 38) 

            if self.args.avg_pool:
                x = [F.avg_pool1d(i, i.size(2)).squeeze(2) for i in x] 
                                                    #[(N,Co), ...]*len(Ks)
            else:
                x = [F.max_pool1d(i, i.size(2)).squeeze(2) for i in x] 
                                                    #[(N,Co), ...]*len(Ks)

            x = torch.cat(x, 1)

            return x 

class LSTM(nn.Module):
	def __init__(self, args, embeddings):
		super(LSTM, self).__init__()

		self.args = args
		
		V = args.embed_num
		D = args.embed_dim

		# initialize embedding layer 
		self.embed = nn.Embedding(V, D)
		self.embed.weight.data = torch.from_numpy(embeddings)
		self.embed.weight.requires_grad=False
		
		self.lstm = nn.LSTM(input_size=D,
				    hidden_size=args.hidden_size,
				    num_layers=1,
				    batch_first=True)
		

	def forward(self, x):
		x = self.embed(x) # (N,W,D) 
 
		hidden = (autograd.Variable(torch.zeros(1, len(x), self.args.hidden_size)),
			autograd.Variable(torch.zeros(1, len(x), self.args.hidden_size)))
		
		out, hidden = self.lstm(x, hidden)
		
		out = out.permute(0,2,1)		# swap axes
		
		return F.max_pool1d(out, out.size(2)).squeeze(2)

class GradReverse(autograd.Function):
    def __init__(self, lambd):
        self.lambd = lambd

    def forward(self, x):
        return x.view_as(x)

    def backward(self, grad_output):
        return (grad_output * -self.lambd)

def grad_reverse(x, lambd):
    return GradReverse(lambd)(x)

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
        #TODO: add leaky parameter [current default neg_slope: 0.01]
        x = F.leaky_relu(x)
        x = self.fc2(x)
        x = self.softmax(x)
        return x 
