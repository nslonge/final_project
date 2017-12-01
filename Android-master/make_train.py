
d_ids = {}
t_ids = {}

fp = open('dev.txt','rb')
for ln in fp:
    ln = ln.split('\t')
    d_ids[int(ln[0])] = 1
fp.close()
fp = open('test.txt','rb')
for ln in fp:
    ln = ln.split('\t')
    t_ids[int(ln[0])] = 1
fp.close()

fp = open('corpus.tsv', 'rb')
fp2 = open('train.txt', 'wb')

for ln in fp:
    id = int(ln.split('\t')[0])
    if not id in d_ids or ln in t_ids:
        ln = str(id) + '\t' + '0' + '\t' + '0' + '\n'
        fp2.write(ln)
fp.close()
fp2.close()

