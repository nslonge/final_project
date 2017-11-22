# -*- coding: utf-8 -*-
"""
Created on Mon Nov 20 15:38:16 2017

@author: Administrator
"""

def calc_eval(dname, fname, dataset, f):
    BM25 = []
    for line in dataset:
        comp = line.split('\t')
        pos = comp[1].split()
        whole = comp[2].split()
        if len(pos) == 0:
            continue        
        BM25.append(f(whole, pos))
    print(fname + ' for ' + dname + ': {:.2%}'.format(float(sum(BM25))/len(BM25)))

def calc_map(whole, pos):
    n_retrieved_pos = 0
    n_retrieved = 0
    p_at_n = []
    for i in range(len(whole)):
        n_retrieved += 1
        if whole[i] in pos:
            n_retrieved_pos += 1
            p_at_n.append(float(n_retrieved_pos)/n_retrieved)
        if n_retrieved_pos == len(pos):
            break
    return float(sum(p_at_n))/len(p_at_n)    
    
def calc_mrr(whole, pos):
    return 1.0 / (whole.index(pos[0]) + 1)

def calc_P1(whole, pos):
    return whole[0] in pos

def calc_P5(whole, pos):
    p5 = [s in pos for s in whole[:5]]
    return float(sum(p5))/len(p5)

def show_eval(name, f):    
    dev = open('askubuntu-master/dev.txt')
    test = open('askubuntu-master/test.txt')   
    print(name + ':')
    calc_eval('dev', name, dev, f)
    calc_eval('test', name, test, f)
    dev.close()
    test.close()

# Make sure that these figures are consistent as reported in the paper
show_eval('MAP', calc_map)
show_eval('MRR', calc_mrr)
show_eval('P1', calc_P1)
show_eval('P5', calc_P5)