from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from nltk.corpus import stopwords
import numpy as np
import pdb
import evaluate

class Dataset:
    def __init__(self, text_path, path, name):
        self.text_path = text_path
        self.path = path
        self.name = name

        # load documents
        self.docs, self.idx2id, self.id2idx = self.load_docs()

        # split documents into train, non_train
        self.train, self.dev = self.split_docs()

        # initialize vectorizer
        self.vectorizer = TfidfVectorizer(ngram_range=(1,1),
                                          stop_words=stopwords.words('english'))
        self.vectorizer.fit(self.train)

        # load mapping from id to pos,neg
        self.id2posneg = self.load_posneg()        

    def id2str(self, id):
        return self.docs[self.id2idx[id]]

    def load_docs(self):
        # load all questions 
        docs = []
        idx2id = {}
        id2idx = {}
        # load documents
        fp = open(self.text_path, 'rb')
        for i, ln in enumerate(fp):
            ln = ln.strip().replace('\n','').split('\t')
            id = int(ln[0])
            doc = ln[1]
            idx2id[i] = id
            id2idx[id] = i
            docs.append(doc)
        fp.close()
        return docs, idx2id, id2idx
            
    def split_docs(self):
        # get list of train and non_train docs
        d_ids = {}
        fp = open(self.path+'{}.txt'.format(self.name))
        for ln in fp:
            id = int(ln.split('\t')[0])
            if not id in d_ids: d_ids[id] =1
        fp.close()

        # split docs
        tr_docs, d_docs = [],[]
        d_id2idx = {}
        for i, doc in enumerate(self.docs):
            id = self.idx2id[i]
            if id in d_ids:
                d_docs.append(doc)
            else:
                tr_docs.append(doc)
        return tr_docs, d_docs

    def load_posneg(self):
        # return dictionary id -> (pos_ids, neg_ids)
        fp = open(self.path + '{}.txt'.format(self.name), 'rb')
        id2posneg = {}
        for ln in fp:
            ln = ln.strip().replace('\n','').split('\t')
            id = int(ln[0])
            pos,neg = ln[1], ln[2]
            pos = map(lambda x: int(x), pos.split())
            neg = map(lambda x: int(x), neg.split())
            neg = filter(lambda x: not x in pos, neg)
            id2posneg[id] = (pos,neg)
        fp.close()
        return id2posneg

def get_scores(data):
    # load samples 
    # create tfidf representation
    
    # get cosine similarity
    MAPs, MRRs, P1s, P5s = [],[],[],[]
    for id,(pos,neg) in data.id2posneg.iteritems():
        # total number of positive examples
        pn = len(pos)
        if pn == 0: continue
       
        # get query index and vector
        q = data.vectorizer.transform([data.id2str(id)])[0]

        # get positive and negative vectors    
        pos = [data.id2str(i) for i in pos]
        neg = [data.id2str(i) for i in neg]
        posneg = data.vectorizer.transform(pos+neg) 

        # compute cosine similarity
        cos_sim = cosine_similarity(q, posneg).tolist()[0]

        # get scores
        cos_sim = zip(cos_sim, 
                      [int(i<pn) for i in range(len(cos_sim))], 
                      [None for i in range(len(cos_sim))])
        cos_sim = (sorted(cos_sim, key=lambda x: x[0], reverse=True), pn)
        MAPs.append(evaluate.MAP(cos_sim))
        MRRs.append(evaluate.MRR(cos_sim))
        P1s.append(evaluate.P(cos_sim,1))
        P5s.append(evaluate.P(cos_sim,5))

    avg = lambda x: sum(x)/float(len(x))
    return avg(MAPs), avg(MRRs), avg(P1s), avg(P5s)

def main():
    # paths
    andr_text = 'Android-master/corpus.txt'
    andr_path = 'Android-master/'
    aub_text = 'askubuntu-master/corpus.txt'
    aub_path = 'askubuntu-master/' 

    # initalize dataset
    dev_data = Dataset(aub_text, aub_path, 'dev')
    test_data = Dataset(aub_text, aub_path, 'test')

    # get dev scores
    MAP,MRR,P1,P5 = get_scores(dev_data)
    print('Dev set scores:')
    print('MAP: {}\nMRR: {}\nP@1: {}\nP@5: {}\n'.format(MAP,MRR,P1,P5))

    # get test scores
    MAP,MRR,P1,P5 = get_scores(test_data)
    print('Test set scores:')
    print('MAP: {}\nMRR: {}\nP@1: {}\nP@5: {}\n'.format(MAP,MRR,P1,P5))

if __name__=="__main__":
    main()
