import os
import sys
import torch
import torch.nn as nn
import torch.autograd as autograd
import torch.nn.functional as F
import torch.utils.data as data
from tqdm import tqdm
import datetime
import pdb
import numpy as np


def MAP(scores):
	# scores: ([(score, is_pos, orig_idx), ...], tot_pos) 
	scores, tot_pos = scores
	cor = 0
	ap = 0
	for i, (score, is_pos, _) in enumerate(scores):
		cor+=is_pos
		precision = cor/float(i+1)
		ap += precision*is_pos
	return ap/float(tot_pos) 

def P(scores, n):
	# scores: ([(score, is_pos, orig_idx), ...], tot_pos) 
	scores, _ = scores
	return sum(map(lambda x: x[1], scores[:n]))/float(n)

def MRR(scores):
	# scores: ([(score, is_pos, orig_idx), ...], tot_pos) 
	scores, _ = scores
	i = 0
	while True:
		if scores[i][1] <> 1:
			i+=1
		else:
			return 1/float(i+1)

def score(s_s, pos_idxs):
	s_s = s_s.data.numpy().T.tolist()
	idxs = [i for i in range(len(s_s[0]))]
	scores = [(sorted(zip(score, p_idxs, idxs), 
					  key=lambda x:x[0], 
					  reverse=True), 
			   sum(p_idxs)) 
			  for (score,p_idxs) in zip(s_s,pos_idxs)]
	#[((score, is_pos, orig_idx), tot_pos), ...]*bs
	MAPs = map(lambda x: MAP(x), scores)
	MRRs = map(lambda x: MRR(x), scores)
	P1s = map(lambda x: P(x, 1), scores)
	P5s = map(lambda x: P(x, 5), scores)
	avg = lambda x: sum(x)/float(len(x))
	return avg(MAPs), avg(MRRs), avg(P1s), avg(P5s)

def get_sample(idx_to_cand, idx_to_vec, ids, titles):
	pos_batch = []
	neg_batch = []
	new_titles = []
	for id, title in zip(ids,titles):
		title = title.numpy().tolist()
		pos, neg = idx_to_cand[id]
		if pos == []:
			continue
		pos_idx = [int(i in pos) for i in neg]
		neg = map(lambda x: idx_to_vec[x].numpy().tolist(), neg)
		pos_batch.append(pos_idx)
		neg_batch.append(neg)
		new_titles.append(title)

	neg_batch = np.asarray(neg_batch)
	neg_batch = np.swapaxes(neg_batch,0,1)

	return torch.LongTensor(new_titles),\
		   pos_batch,\
		   torch.LongTensor(neg_batch)

def evaluate(model, data, args):
	#load single mini-batch
	data_loader = torch.utils.data.DataLoader(
		data,
		batch_size=data.__len__(),
		shuffle=False,
		num_workers=4,
		drop_last=False)

	# train on each batch
	s_s,pos = None, None
	for batch in tqdm(data_loader):
		# first, get additional vectors per batch
		ids = batch['id']
		titles = batch['title']

		# for each id, look up associated questions
		titles, pos, neg = get_sample(data.idx_to_cand,
									  data.idx_to_vec, 
									  ids, titles)

		q = autograd.Variable(titles)
		ps = autograd.Variable(neg)
			
		# run the batch through the model
		q = model(q)
		
		ps = ps.contiguous().view(-1,38)
		ps = model(ps)
	
		if args.model == 'cnn':
			ps = ps.contiguous().view(args.neg_samples,-1,
									  len(args.kernel_sizes) * args.kernel_num)
		elif args.model == 'lstm':
			ps = ps.contiguous().view(args.neg_samples,-1,args.hidden_size)
		
		# get cosine similarities	
		qs = q.repeat(args.neg_samples,1,1)
		cos2 = nn.CosineSimilarity(dim=2)
		s_s = cos2(qs,ps) 

	map, mrr, p1, p5 = score(s_s, pos)
	print('MAP: {}\nMRR: {}\nP@1: {}\nP@5: {}\n'.format(map,mrr,p1,p5))


