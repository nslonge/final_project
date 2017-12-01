import gzip
import pdb

vocab = {}

with gzip.open('corpus.tsv.gz') as fp:
    for ln in fp:
        ln = ln.replace('\n','').split('\t')
        ln = ln[1] + ln[2]
        ln = ln.split()
        for word in ln:
            if not word in vocab: vocab[word]=1

embeddings = []

with open('glove.840B.300d.txt') as fp:
    for ln in fp:
        word = ln.split()[0]
        if word in vocab:
            embeddings.append(ln)

fp = open('new_embeddings.txt', 'wb')
for ln in embeddings:
    fp.write(ln)
fp.close()
