import os
import sys
import torch
import torch.nn as nn
import torch.autograd as autograd
import torch.nn.functional as F
import torch.utils.data as data
from tqdm import tqdm
from sklearn.metrics import precision_score, recall_score, accuracy_score
import datetime
import pdb
import numpy as np


def MAP(scores):
	# scores: ([(score, is_pos, orig_idx), ...], tot_pos) 
	scores, tot_pos = scores
	cor = 0
	ap = 0
	for i, (score, is_pos, _) in enumerate(scores):
		if score <> 0: cor+=is_pos
		precision = cor/float(i+1)
		ap += precision*is_pos
	return ap/float(tot_pos) 

def P(scores, n):
	# scores: ([(score, is_pos, orig_idx), ...], tot_pos) 
	scores, tot_pos = scores
        total = 0
        for score in scores[:n]:
            if score[0] <> 0: total+=score[1]
        return total/float(n)
        #return sum(map(lambda x: x[1], scores[:m]))/float(m)            

def MRR(scores):
	# scores: ([(score, is_pos, orig_idx), ...], tot_pos) 
	scores, _ = scores
	i = 0
	while True:
	    if scores[i][1] <> 1:
	    	i+=1
	    else:
                if scores[i][0] == 0.0: return 0
	    	return 1/float(i+1)

def score(s_s, pos_idxs):
	s_s = s_s.data.numpy().T.tolist()
	idxs = [i for i in range(len(s_s[0]))]
	scores = [(sorted(zip(score, p_idxs, idxs), 
			  key=lambda x:x[0], 
			  reverse=True), 
		    sum(p_idxs)) for (score,p_idxs) in zip(s_s,pos_idxs)]
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
            neg = map(lambda x: idx_to_vec(x).numpy().tolist(), neg)
            pos_batch.append(pos_idx)
            neg_batch.append(neg)
            new_titles.append(title)

	neg_batch = np.asarray(neg_batch)
	neg_batch = np.swapaxes(neg_batch,0,1)

	return torch.LongTensor(new_titles),\
		   pos_batch,\
		   torch.LongTensor(neg_batch)

def d_evaluate(model, s_data, t_data):
    source_data = torch.utils.data.DataLoader(
                    s_data,
                    batch_size=s_data.__len__(),
                    shuffle=False,
                    num_workers=4,
                    drop_last=False)

    target_data = torch.utils.data.DataLoader(
                    t_data,
                    batch_size=t_data.__len__(),
                    shuffle=False,
                    num_workers=4,
                    drop_last=False)

    precision = 0
    recall = 0
    accuracy = 0

    for s_batch, t_batch in tqdm(zip(source_data,target_data)):
        s_titles = s_batch['title']
        t_titles = t_batch['title']
        x = torch.cat([s_titles, t_titles], 0)
        x = autograd.Variable(x)

        y1 = np.ones((s_titles.shape[0]))
        y0 = np.zeros((t_titles.shape[0]))
        y = np.concatenate([y1, y0])
      
        pdb.set_trace() 
        out = model(x)
        out = out.data.topk(1).numpy()

        pdb.set_trace()

        precision = precision_score(y, out)        
        recall = recall_score(y, out)        
        accuracy = accuracy_score(y, out)        

    print('Domain Classification:\nPrecision: {}\nRecall: {}\nAccuracy: {}\n'.format(precision, recall, accuracy))
 

def q_evaluate(model, data, args):
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
                                      lambda x: data.idx_to_vec[x][0], 
                                      ids, titles)

        q = autograd.Variable(titles)
        ps = autograd.Variable(neg)
                
        # run the batch through the model
        q = model(q)
        
        ps = ps.contiguous().view(-1,args.max_title)
        ps = model(ps)

        if args.model == 'cnn':
                ps = ps.contiguous().view(args.neg_samples,-1,
                                len(args.kernel_sizes)*args.kernel_num)
        elif args.model == 'lstm':
                ps = ps.contiguous().view(args.neg_samples,-1,
                                          args.hidden_size)

        # if needed, run computation on body
        if args.use_body:
            bodies = batch['body']
            bodies, pos, neg = get_sample(data.idx_to_cand,
                                       lambda x: data.idx_to_vec[x][1], 
                                       ids, bodies)

            q_b = autograd.Variable(bodies)
            ps_b = autograd.Variable(neg)

            q_b = model(q_b)
            ps_b = ps_b.contiguous().view(-1,args.max_body)
            ps_b = model(ps_b)
            
            if args.model == 'cnn':
                    ps = ps.contiguous().view(args.neg_samples+1,-1,
                                len(args.kernel_sizes)*args.kernel_num)
            elif args.model == 'lstm':
                    ps = ps.contiguous().view(args.neg_samples+1,-1,
                                              args.hidden_size)
            q = (q+q_b)/2.0
            p_plus = (p_plus+p_plus_b)/2.0
            ps = (ps+ps_b)/2.0

        # get cosine similarities	
        qs = q.repeat(args.neg_samples,1,1)
        cos2 = nn.CosineSimilarity(dim=2)
        s_s = cos2(qs,ps) 

    map, mrr, p1, p5 = score(s_s, pos)
    print('Similarity Task:\nMAP: {}\nMRR: {}\nP@1: {}\nP@5: {}\n'.format(map,mrr,p1,p5))


