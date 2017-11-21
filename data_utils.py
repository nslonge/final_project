import gzip
import numpy as np
import torch
import torch.utils.data as data
import cPickle as pickle
import tqdm
import pdb
import collections

PATH="askubuntu-master/text_tokenized.txt.gz"
PATH2 = "askubuntu-master/{}_random.txt"

torch.manual_seed(1)
#inherit from torch Dataset class, so i can make batches
class FullDataset(data.Dataset):
	def __init__(self, name, word_to_indx, embeddings, args):
		self.name = name
		self.dataset = []
		self.word_to_indx  = word_to_indx
		self.max_length = 38
		self.idx_to_vec = {}

		self.idx_to_cand = self.load_cand_sets(args,name)

		with gzip.open(PATH) as file:
			lines = file.readlines()
			for line in tqdm.tqdm(lines):
				sample = self.processLine(line,embeddings)
				if sample <> None:
					self.dataset.append(sample)
			file.close()

	def load_cand_sets(self, args, name):
		if name == 'train':
			path = PATH2.format(self.name)
		elif name == 'dev':
			path = 'askubuntu-master/dev.txt'
		elif name == 'test':
			path = 'askubuntu-master/test.txt'

		idx_to_cand = {}
		with open(path) as file:
			lines = file.readlines()
			for line in lines:#[:1000]:
				line = line.split('\t')
				idx = int(line[0])
				pos = map(lambda x: int(x), line[1].split())
				neg = map(lambda x: int(x), line[2].split())
				#neg = filter(lambda x: x <> pos, neg)
				neg = neg[:args.neg_samples]
				idx_to_cand[idx] = (pos,neg)
			file.close()
		return idx_to_cand
			
	## Convert one line from dataset to {Text, Tensor, Labels}
	def processLine(self, line, embeddings):
		line = line.split('\t')
		id = int(line[0])
		title = line[1].split()
		#body = line[2].split()

		x =  getIndicesTensor(title, self.word_to_indx, self.max_length)
		sample = {'id':id, 'title':x}
		self.idx_to_vec[id] = x
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
	text_indx = [word_to_indx[x] if x in word_to_indx 
								 else nil_indx for x in text_arr]
	if len(text_indx) < max_length:
		text_indx.extend([nil_indx for _ in range(max_length-len(text_indx))])

	x =  torch.LongTensor(text_indx)

	return x

# load embedding for each word
def getEmbeddingTensor():
	embedding_path='askubuntu-master/vector/vectors_pruned.200.txt.gz'
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
def load_dataset(args):
	print("\nLoading data...")
	embeddings, word_to_indx = getEmbeddingTensor()

	# load questions
	train_data = FullDataset('train', word_to_indx,embeddings, args)
	dev_data = FullDataset('dev', word_to_indx, embeddings, args)
	test_data = FullDataset('test', word_to_indx, embeddings, args)
	return train_data, dev_data, test_data, embeddings

