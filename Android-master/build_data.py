name = 'test'

neg, pos = {},{}
fp = open('{}.neg.txt'.format(name), 'rb')
for ln in fp:
    ln = ln.strip().replace('\n','').split()
    id = int(ln[0])
    qn = ln[1]
    if id in neg: neg[id].append(qn)
    else: neg[id] = [qn]
fp.close()
fp = open('{}.pos.txt'.format(name), 'rb')
for ln in fp:
    ln = ln.strip().replace('\n','').split()
    id = int(ln[0])
    qp = ln[1]
    if id in pos: pos[id].append(qp)
    else: pos[id] = [qp]
fp.close()
fp = open('{}.txt'.format(name), 'wb')
for id in sorted(pos.iterkeys(), key=lambda x: int(x)):
    qp = pos[id]
    if not id in neg: print 'error'; break
    qn = neg[id]
    fp.write(str(id) + '\t' + ' '.join(qp) + '\t' + ' '.join(qp + qn) + '\n')
fp.close()
