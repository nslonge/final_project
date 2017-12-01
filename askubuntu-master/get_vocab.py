import gzip
import pdb

vocab = {}

with gzip.open('corpus.txt.gz') as fp:
    for ln in fp:
        ln = ln.replace('\n','').split('\t')
        ln = ln[1] + ln[2]
        ln = ln.split()
        for word in ln:
            if not word in vocab: vocab[word]=1
    fp.close()
with gzip.open('corpus2.txt.gz') as fp:
    for ln in fp:
        ln = ln.replace('\n','').split('\t')
        ln = ln[1] + ln[2]
        ln = ln.split()
        for word in ln:
            if not word in vocab: vocab[word]=1
    fp.close()

embeddings = []

with open('glove.840B.300d.txt') as fp:
    for ln in fp:
        word = ln.split()[0]
        if word in vocab:
            embeddings.append(ln)

fp = open('vector/vectors_pruned.300.txt', 'wb')
for ln in embeddings:
    fp.write(ln)
fp.close()
