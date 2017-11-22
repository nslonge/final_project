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

def train_model(train_data, dev_data, test_data, model, args, only_eval=False):
	# use an optimized version of SGD
	parameters = filter(lambda p: p.requires_grad, model.parameters())
	optimizer = torch.optim.Adam(parameters,
								 lr=args.lr)
	
	for epoch in range(1, args.epochs+1):
		print("-------------\nEpoch {}:\n".format(epoch))

		# train
		if not only_eval:
			train_loss = run_epoch(train_data, True, model, optimizer, args)
			print('/nTrain loss: {}'.format(train_loss))

		dev_eval = run_epoch(dev_data, False, model, optimizer, args)
		test_eval = run_epoch(test_data, False, model, optimizer, args)
	

def get_pos_neg(idx_to_cand, idx_to_vec, ids, titles, is_training):
	pos_batch = []
	neg_batch = []
	new_titles = []
	pos_index = []
	best_index = []
	for id, title in zip(ids,titles):
		title = title.numpy().tolist()
		pos, neg = idx_to_cand[id]
		pos_index.append([n in pos for n in neg])
		best_index.append(neg.index(pos[0]) if len(pos) > 0 and pos[0] in neg else 0)
		neg = map(lambda x: idx_to_vec[x].numpy().tolist(), neg)
		if len(pos) == 0:
			continue	
		for p in pos:
			p = idx_to_vec[p].numpy().tolist()
			tmp = list(neg)
			tmp.insert(0,p)
			pos_batch.append(p)
			neg_batch.append(tmp)
			new_titles.append(title)
			if not is_training:		# for eval, we just need once
				break

	neg_batch = np.asarray(neg_batch)
	neg_batch = np.swapaxes(neg_batch,0,1)

	return torch.LongTensor(new_titles),\
		   torch.LongTensor(pos_batch),\
		   torch.LongTensor(neg_batch),\
		   torch.IntTensor(pos_index),\
		   torch.IntTensor(best_index)

		   
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

def calc_sims(q, ps, args):
	#q: (bs,Co), get [s(q,p_i),...]*neg_samples
	qs = q.repeat(args.neg_samples+1,1,1) # (neg_samples,bs)
	cos2 = nn.CosineSimilarity(dim=2)
	return cos2(qs,ps)

def calc_map(sims, idx):
	pairs = [(sims[i], idx[i]) for i in range(len(idx))]
	pairs = sorted(pairs, key = lambda x: x[0], reverse = True)
	n_retrieved_pos = 0
	n_retrieved = 0
	p_at_n = []
	for i in range(len(pairs)):
		n_retrieved += 1
		if pairs[i][1]:
			n_retrieved_pos += 1
			p_at_n.append(float(n_retrieved_pos)/n_retrieved)
		if n_retrieved_pos == sum(idx):
			break
	return float(sum(p_at_n))/len(p_at_n)    

def calc_mrr(sims, best_pos):
	pairs = [(sims[i], i) for i in range(len(sims))]
	pairs = sorted(pairs, key = lambda x: x[0], reverse = True)
	return 1.0 / (1 + [p[1] for p in pairs].index(best_pos))

def calc_p1(sims, idx):
	pairs = [(sims[i], idx[i]) for i in range(len(idx))]
	pairs = sorted(pairs, key = lambda x: x[0], reverse = True)
	return pairs[0][1]

def calc_p5(sims, idx):
	pairs = [(sims[i], idx[i]) for i in range(len(idx))]
	pairs = sorted(pairs, key = lambda x: x[0], reverse = True)
	return sum([p[1] for p in pairs[0:5]]) / 5.0

def run_epoch(data, is_training, model, optimizer, args):
	# load random batches
	data_loader = torch.utils.data.DataLoader(
		data,
		batch_size=args.batch_size,
		shuffle=True,
		num_workers=4,
		drop_last=True)

	losses = []
	evals = []
	
	# train on each batch
	for batch in tqdm(data_loader):
		# first, get additional vectors per batch
		ids = batch['id']
		titles = batch['title']

		# for each id, look up associated questions
		titles, pos, neg, idx, best_pos = get_pos_neg(data.idx_to_cand,
									   data.idx_to_vec, 
									   ids, titles, is_training)

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
			ps = ps.contiguous().view(args.neg_samples+1,
								  -1,
								  len(args.kernel_sizes) * args.kernel_num)
		elif args.model == 'lstm':
			ps = ps.contiguous().view(args.neg_samples+1,-1,args.hidden_size)

		if is_training:
			loss = mmloss(q, p_plus, ps, args)
			# back-propegate to compute gradient
			loss.backward()
			# descend along gradient
			optimizer.step()
			losses.append(loss.cpu().data[0])
		else:
			s_s = calc_sims(q, ps, args)
			similarities = s_s.data.numpy().T
			for k in range(len(similarities)):
				s_k, i_k = similarities[k][1:], idx[k].numpy().tolist()
				if sum(i_k) == 0:
					continue
				evals.append((calc_map(s_k, i_k),\
				  calc_mrr(s_k, best_pos[k]),\
				  calc_p1(s_k, i_k),\
				  calc_p5(s_k, i_k),\
				  ))			
	
	if not is_training:
		eval_res = [float(sum([e[i] for e in evals])) / len(evals) for i in range(len(evals[0]))]
		print 'MAP \t MRR \t P@1 \t P@5'
		print eval_res
		return eval_res
	else:
		avg_loss = np.mean(losses)
		return avg_loss