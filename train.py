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
import evaluate

torch.manual_seed(1)

def train_model(train_data, dev_data, test_data, model, args):
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

		print('\nEvaluating on dev')
		evaluate.evaluate(model, dev_data, args)

		print('Evaluating on test')
		evaluate.evaluate(model, test_data, args)

	

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

def mmloss(q, p_plus, ps, args):
	#q: (bs,Co), p_plus: (bs, Co), ps: (neg_samples, bs, Co)	

	# get s(q,p+)
	cos = nn.CosineSimilarity()
	s_0 = cos(q, p_plus) # (bs)

	# get [s(q,p_i),...]*neg_samples
	qs = q.repeat(args.neg_samples+1,1,1) # (neg_samples,bs)
	cos2 = nn.CosineSimilarity(dim=2)
	s_s = cos2(qs,ps) # (neg_samples, bs)

	# compute delta tensor 
	bs = s_0.data.shape[0]
	delta = np.array([0.0]+[args.delta for _ in range(args.neg_samples)])
	delta = np.tile(delta, (bs,1))
	delta = delta.T #(neg_samples, bs)
	delta = autograd.Variable(torch.Tensor(delta))

	# compute score
	s_0 = s_0.repeat(args.neg_samples+1,1) # (neg_samples, bs)
	scores = s_s-s_0+delta # (neg_samples, bs)
	score,_ = torch.max(scores,0)
	return torch.mean(score)

def run_epoch(data, is_training, model, optimizer, args):
	# load random batches
	data_loader = torch.utils.data.DataLoader(
		data,
		batch_size=args.batch_size,
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

			
		# zero all gradients
		optimizer.zero_grad()

		# run the batch through the model
		q = model(q)
		p_plus = model(p_plus)
		ps = ps.contiguous().view(-1,38)
		ps = model(ps)
		
		if args.model == 'cnn':
			ps = ps.contiguous().view(args.neg_samples+1,
									  -1,
									  len(args.kernel_sizes) * args.kernel_num)
		elif args.model == 'lstm':
			ps = ps.contiguous().view(args.neg_samples+1,-1,args.hidden_size)
			
		loss = mmloss(q, p_plus, ps, args)			

		# back-propegate to compute gradient
		loss.backward()
		# descend along gradient
		optimizer.step()

		losses.append(loss.cpu().data[0])

	# Calculate epoch level scores
	avg_loss = np.mean(losses)
	return avg_loss
