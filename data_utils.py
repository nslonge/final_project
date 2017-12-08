import gzip
import numpy as np
import torch
import torch.utils.data as data
import cPickle as pickle
import tqdm
import pdb
import collections

torch.manual_seed(1)
#inherit from torch Dataset class, so i can make batches
class FullDataset(data.Dataset):
	def __init__(self, name, data_set, word_to_indx, embeddings, args):
            self.name = name
            self.path = '{}/{}.txt'.format(data_set,name)	
            self.dataset = []
            self.word_to_indx  = word_to_indx
            self.idx_to_vec = {}

            self.idx_to_cand = self.load_cand_sets(args)

            with gzip.open('{}/corpus.txt.gz'.format(data_set)) as file:
                lines = file.readlines()
                for line in tqdm.tqdm(lines):
                        sample = self.processLine(line,embeddings,args)
                        if sample <> None:
                                self.dataset.append(sample)
                file.close()

	def load_cand_sets(self, args):
            idx_to_cand = {}
            with open(self.path) as file:
                lines = file.readlines()
                for line in lines:#[:1000]:
                    line = line.split('\t')
                    idx = int(line[0])
                    pos = map(lambda x: int(x), line[1].split())
                    neg = map(lambda x: int(x), line[2].split())
                    mx_samples = 100
                    if self.name == 'train':
                        mx_samples = args.neg_samples
                    neg = neg[:mx_samples]
                    idx_to_cand[idx] = (pos,neg)
                file.close()
            return idx_to_cand
			
	## Convert one line from dataset to {Text, Tensor, Labels}
	def processLine(self, line, embeddings, args):
		line = line.split('\t')
		id = int(line[0])
		title = line[1].split()
		x =  getIndicesTensor(title, self.word_to_indx, args.max_title)
                sample = {'id':id, 'title':x}

                # if needed, load body
                y = None
                if args.use_body:
		    body = line[2].split()
		    y =  getIndicesTensor(body, self.word_to_indx, 
                                          args.max_body)
		    sample = {'id':id, 'title':x, 'body':y}
                self.idx_to_vec[id] = (x,y)
		if not id in self.idx_to_cand:
			return None
		return sample

	def __len__(self):
		return len(self.dataset)

	def __getitem__(self,index):
		sample = self.dataset[index]
		return sample

# load each document into 1 x max_length tensor, with word index entries
def getIndicesTensor(text_arr, word_to_indx, max_length):
	nil_indx = 0
	text_indx = [word_to_indx[x] if x in word_to_indx else nil_indx for x in text_arr][:max_length]
	if len(text_indx) < max_length:
		text_indx.extend([nil_indx for _ in range(max_length-len(text_indx))])

	x =  torch.LongTensor(text_indx)

	return x

# load embedding for each word
def getEmbeddingTensor(args):
	embedding_path='askubuntu-master/vector/vectors_pruned.{}.txt.gz'.format(args.embed_dim)
	lines = []
	with gzip.open(embedding_path) as file:
		lines = file.readlines()
		file.close()
	embedding_tensor = []
	word_to_indx = {}
	for indx, l in enumerate(lines):
		word, emb = l.split()[0], l.split()[1:]
		vector = [float(x) for x in emb ]
		# This is for the 'nil_index' words that aren't part
		# of the embedding dictionary
		if indx == 0:
			embedding_tensor.append( np.zeros( len(vector) ) )
		embedding_tensor.append(vector)
		word_to_indx[word] = indx+1
	embedding_tensor = np.array(embedding_tensor, dtype=np.float32)

	return embedding_tensor, word_to_indx


# Build dataset
def load_dataset(args, data_set, dtrain=False):
    print("\nLoading data...")
    embeddings, word_to_indx = getEmbeddingTensor(args)

    # load questions
    train_data = FullDataset('train',data_set,word_to_indx,embeddings,args)
    dev_data = FullDataset('dev',data_set,word_to_indx,embeddings,args)
    test_data = FullDataset('test',data_set,word_to_indx,embeddings,args)
    if not dtrain:
        return train_data, dev_data, test_data, embeddings
    else: 
        dtrain_data = FullDataset('d_train', data_set, word_to_indx, embeddings, args)
    return train_data, dtrain_data, dev_data, test_data, embeddings   


