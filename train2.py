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

def train_model(s_train, sd_train, s_dev, s_test, 
                t_train, t_dev, t_test,
                q_model, d_model, args):

    q_parameters = filter(lambda p: p.requires_grad, q_model.parameters())
    d_parameters = filter(lambda p: p.requires_grad, d_model.parameters())

    optimizer = None
    q_opt = torch.optim.Adam(q_parameters,lr=args.lr)
    d_opt = torch.optim.Adam(d_parameters,lr=-args.lr_d)

    scores = []
    for epoch in range(1, args.epochs+1):
        print("-------------\nEpoch {}:\n".format(epoch))

        # train
        loss1,loss2 = run_epoch(s_train, sd_train, t_train, 
                                q_model, d_model, 
                                q_opt, d_opt, args)

        print('\nTrain loss: {}, {}'.format(loss1, loss2))
        torch.save(q_model, args.save_path + args.name + '.' +str(epoch) + '.pkl')

#		print('\nEvaluating on source dev')
#		evaluate.q_evaluate(q_model, s_dev, args)

#		print('Evaluating on source test')
#		evaluate.q_evaluate(q_model, s_test, args)

        print('\nEvaluating on target dev')
        evaluate.q_evaluate(q_model, t_dev, args)

        print('Evaluating on target test')
        evaluate.q_evaluate(q_model, t_test, args)

        if args.full_eval and not args.use_mmd:
            print('Evaluating domain classifier on dev')
            evaluate.d_evaluate(q_model, d_model, s_dev, t_dev)

            print('Evaluating domain classifier on test')
            evaluate.d_evaluate(q_model, d_model, s_test, t_test)

        

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


def calc_mmd_loss(bottleneck, y):
	'''
	Given bottleneck layer representations for sample points from S and T,
	with y as the domain indicator, calculate the Maximum Mean Discrepancy
	value of the two distributions
	
	Indicator y: 1 for source, 0 for target
	'''
	y = y.type(torch.ByteTensor)
	
	repr_len = bottleneck.size()[1]
	# (bs, repr_len)
	phi_s = torch.t(torch.t(bottleneck).masked_select(y).view(repr_len,-1))
	phi_t = torch.t(torch.t(bottleneck).masked_select(y^1).view(repr_len,-1))
	
	phi_s = torch.mean(phi_s, dim=0, keepdim=True)	# (1, repr_len)
	phi_t = torch.mean(phi_t, dim=0, keepdim=True)
	
	#TODO: parameterize p-norm in args
	pdist = nn.PairwiseDistance(p=2)
	mmd = pdist(phi_s, phi_t)
	return mmd

def calc_mmd_loss2(bottleneck, n, m):
    source, target = torch.split(bottleneck, n, 0) 
    source = torch.mean(source, 0)
    target = torch.mean(target, 0)
    return torch.norm(source-target, p=2)
	
def run_epoch(s_data, sd_data, t_data, q_model, d_model, q_opt, d_opt, args):
	# load random batches
	source_query_data = torch.utils.data.DataLoader(
		s_data,
		batch_size=args.batch_size,
		shuffle=True,
		num_workers=4,
		drop_last=True)
	domain_bs = t_data.__len__()/(s_data.__len__()/float(args.batch_size))
	domain_bs = int(domain_bs) + 1
	source_domain_data = torch.utils.data.DataLoader(
		sd_data,
		batch_size=domain_bs,
		shuffle=True,
		num_workers=4,
		drop_last=True)
	target_domain_data = torch.utils.data.DataLoader(
		t_data,
		batch_size=domain_bs,
		shuffle=True,
		num_workers=4,
		drop_last=True)
	source_domain_data = iter(source_domain_data)
	target_domain_data = iter(target_domain_data)

	losses1 = []
	losses2 = []

	criterion = torch.nn.NLLLoss()

	# train on each batch
	for s_batch in tqdm(source_query_data):
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
            
            q_opt.zero_grad()
        
            # run the batch through the model
            if args.use_mmd:
	            q, _ = q_model(q)
	            p_plus, _ = q_model(p_plus)
	            ps = ps.contiguous().view(-1,args.max_title)
	            ps, _ = q_model(ps)
            else:
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

            # update model
            loss1 = mmloss(q, p_plus, ps, args)			
            losses1.append(loss1.cpu().data[0])
           
            # check if we should continue 
            if not args.full_eval: 
                loss1.backward()
                q_opt.step()
                losses2.append(0)
                continue          
                        
            # ---------------------- Domain Classifier -------------------
             
            # sample from source and target domains
            x1 = source_domain_data.next()['title']
            y1 = torch.LongTensor((x1.shape[0])).fill_(1)

            x0 = target_domain_data.next()['title']
            y0 = torch.LongTensor((x0.shape[0])).fill_(0)

            x = autograd.Variable(torch.cat([x1,x0],0))
            y = autograd.Variable(torch.cat([y1,y0],0))
            
            q_opt.zero_grad()
            if not args.use_mmd:
		d_opt.zero_grad()
          
            if args.use_mmd:
                _, bottleneck = q_model(x)
                loss2 = calc_mmd_loss2(bottleneck, x1.shape[0], x0.shape[0])
                loss = loss1 + args.lambda_mmd * loss2
                        
            else:
                out = q_model(x) 
                out = d_model(out)
           
                loss2 = criterion(out,y)			
                loss = loss1 - args.lambd * loss2
				
            loss.backward()
            q_opt.step()
            if not args.use_mmd:
		d_opt.step()
				
            losses2.append(loss2.cpu().data[0])


	# Calculate epoch level scores
	avg_loss1 = np.mean(losses1)
	avg_loss2 = np.mean(losses2)
	return avg_loss1, avg_loss2
