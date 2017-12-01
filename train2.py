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

def train_model(s_train, s_dev, s_test, 
                t_train, t_dev, t_test,
                q_model, d_model, args):

	# use an optimized version of SGD
	q_parameters = filter(lambda p: p.requires_grad, q_model.parameters())
	d_parameters = filter(lambda p: p.requires_grad, d_model.parameters())

        optimizer = None
        q_opt = torch.optim.Adam(q_parameters,lr=args.lr)
        d_opt = torch.optim.Adam(d_parameters,lr=args.lr)

	scores = []
	for epoch in range(1, args.epochs+1):
		print("-------------\nEpoch {}:\n".format(epoch))

		# train
		loss1,loss2 = run_epoch(s_train, t_train, 
                                        q_model, d_model, 
                                        q_opt, d_opt, args)

		print('Train loss: {}, {}'.format(loss1, loss2))
		#torch.save(q_model, args.save_path)	

		print('\nEvaluating on source dev')
		evaluate.q_evaluate(q_model, s_dev, args)

		print('Evaluating on source test')
		evaluate.q_evaluate(q_model, s_test, args)

                print('\nEvaluating on target dev')
		evaluate.q_evaluate(q_model, t_dev, args)

		print('Evaluating on target test')
		evaluate.q_evaluate(q_model, t_test, args)

                print('Evaluating domain classifier on dev')
		evaluate.d_evaluate(q_model, s_dev, t_dev)

                print('Evaluating domain classifier on test')
		evaluate.d_evaluate(q_model, s_test, t_test)


	
def get_pos_neg(idx_to_cand, idx_to_vec, ids, titles):
	pos_batch = []
	neg_batch = []
	new_titles = []
	for id, title in zip(ids,titles):
		title = title.numpy().tolist()
		pos, neg = idx_to_cand[id]
		neg = map(lambda x: idx_to_vec(x).numpy().tolist(), neg)
		for p in pos:
			p = idx_to_vec(p).numpy().tolist()
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
    
def run_epoch(s_data, t_data, q_model, d_model, q_opt, d_opt, args):

	# load random batches
	source_data = torch.utils.data.DataLoader(
		s_data,
		batch_size=args.batch_size,
		shuffle=True,
		num_workers=4,
		drop_last=True)
        target_data = torch.utils.data.DataLoader(
		t_data,
		batch_size=args.batch_size,
		shuffle=True,
		num_workers=4,
		drop_last=True)
        target_data = iter(target_data)

	losses1 = []
	losses2 = []

        criterion = torch.nn.NLLLoss()

	# train on each batch
	for s_batch in tqdm(source_data):
            # load data
            s_ids = s_batch['id']
            s_titles = s_batch['title']

            # -------------- Task Classifier ----------------------------
            titles, pos, neg = get_pos_neg(s_data.idx_to_cand,
                                          lambda x: s_data.idx_to_vec[x][0],
                                           s_ids, s_titles)

            q = autograd.Variable(titles)
            p_plus = autograd.Variable(pos)
            ps = autograd.Variable(neg)
                    
            # run the batch through the model
            q = q_model(q)
            p_plus = q_model(p_plus)
            ps = ps.contiguous().view(-1,args.max_title)
            ps = q_model(ps)
            
            if args.model == 'cnn':
                    ps = ps.contiguous().view(args.neg_samples+1,-1, 
                                    len(args.kernel_sizes)*args.kernel_num)
            elif args.model == 'lstm':
                    ps = ps.contiguous().view(args.neg_samples+1,-1, 
                                              args.hidden_size)

            loss1 = mmloss(q, p_plus, ps, args)			
            loss1.backward()
            q_opt.step()

            # ---------------------- Domain Classifier -------------------
            # fewer target data points, so check if we have one
            try: t_batch = target_data.next()
            except : continue
            
            t_titles = t_batch['title']
            x = torch.cat([s_titles,t_titles],0)
            x = autograd.Variable(x)
            
            y1 = torch.ones((s_titles.shape[0]))
            y2 = torch.zeros((t_titles.shape[0]))
            y = torch.cat([y1,y2],0) 
            y = autograd.Variable(y)           
 
            d_opt.zero_grad()
           
            out = d_model(x)
            loss2 = criterion(x,y)			
            loss2.backward()
            d_opt.step()

            losses1.append(loss1.cpu().data[0])
            losses2.append(loss2.cpu().data[0])

	# Calculate epoch level scores
	avg_loss1 = np.mean(losses1)
	avg_loss2 = np.mean(losses2)
	return avg_loss1, avg_loss2