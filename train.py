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


torch.manual_seed(1)

def train_model(train_data, model, args):
	# use an optimized version of SGD
	parameters = filter(lambda p: p.requires_grad, model.parameters())#
	optimizer = torch.optim.Adam(parameters,
								 lr=args.lr)
	scores = []
	for epoch in range(1, args.epochs+1):
		print("-------------\nEpoch {}:\n".format(epoch))

		# train
		loss = run_epoch(train_data, True, model, optimizer, args)
		#print('Train correct: {} ({}/{})'.format(float(guess)/tot,
		#											 guess,
		#											 tot))
		print('Train loss: {}'.format(loss))
		torch.save(model, args.save_path)	
	

def get_pos_neg(idx_to_cand, idx_to_vec, ids, titles):
	pos_batch = []
	neg_batch = []
	new_titles = []
	for id, title in zip(ids,titles):
		title = title.numpy().tolist()
		pos, neg = idx_to_cand[id]
		neg = map(lambda x: idx_to_vec[x].numpy().tolist(), neg)
		for p in pos:
			p = idx_to_vec[p].numpy().tolist()
			tmp = list(neg)
			tmp.insert(0,p)
			pos_batch.append(p)
			neg_batch.append(tmp)
			new_titles.append(title)

	neg_batch = np.asarray(neg_batch)
	neg_batch = np.swapaxes(neg_batch,0,1)

	return torch.LongTensor(new_titles),\
		   torch.LongTensor(pos_batch),\
		   torch.LongTensor(neg_batch)

def mmloss(q, p_plus, ps, n_size):
	#q: bs x Co
	#p_plus: bs x Co
	#ps: 101 x bs x Co	
	cos = nn.CosineSimilarity()
	s_0 = cos(q, p_plus) # bs x 1

	qs = q.repeat(n_size+1,1,1) # 101 x bs x Co
	cos2 = nn.CosineSimilarity(dim=2)
	s_s = cos2(qs,ps) # 101 x bs x 1
    
    # try to make sure that this line keeps being there

	s_0 = s_0.repeat(n_size+1,1) # 101 x bs x 1
	scores = s_0-s_s # 101 x bs x 1
	score,_ = torch.max(scores,0) 
	return torch.mean(score)

def run_epoch(data, is_training, model, optimizer, args):
	n_size = args.neg_samples

	if is_training: bs = args.batch_size
	
	# load random batches
	data_loader = torch.utils.data.DataLoader(
		data,
		batch_size=bs,
		shuffle=True,
		num_workers=4,
		drop_last=True)

	losses = []

	# train on each batch
	for batch in tqdm(data_loader):
		# first, get additional vectors per batch
		ids = batch['id']
		titles = batch['title']


		# for each id, look up associated questions
		titles, pos, neg = get_pos_neg(data.idx_to_cand,
									   data.idx_to_vec, 
									   ids, titles)

		q = autograd.Variable(titles)
		p_plus = autograd.Variable(pos)
		ps = autograd.Variable(neg)
			
		if is_training:
			# zero all gradients
			optimizer.zero_grad()

		# run the batch through the model
		q = model(q)
		p_plus = model(p_plus)
		ps = ps.contiguous().view(-1,38)
		ps = model(ps)
		
		if args.model == 'cnn':
			ps = ps.contiguous().view(n_size+1,-1,3*args.kernel_num)
		elif args.model == 'lstm':
			ps = ps.contiguous().view(n_size+1,-1,args.hidden_size)
			
		loss = mmloss(q, p_plus, ps, n_size)			

		if is_training:
			# back-propegate to compute gradient
			loss.backward()
			# descend along gradient
			optimizer.step()

		losses.append(loss.cpu().data[0])

	# Calculate epoch level scores
	avg_loss = np.mean(losses)
	return avg_loss
