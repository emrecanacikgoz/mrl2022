
readers= []
readers.append(open('lemma_preds.txt','r+')) 
for i in range(10):
    readers.append( open('tag'+str(i)+'_preds.txt') )
   
preds=dict()
for i,line in enumerate(readers[0]):
    preds[i] =[]
readers[0].seek(0)

with open('/home/mugekural/OpenPrompt/datasets/eng.dev.txt','r+') as f:
    for i,line in enumerate(f):
        snt = line.split('\t')[0]
        preds[i].append(snt)

for i,line in enumerate(readers[0]):
    lemma = line.strip()
    preds[i].append(lemma)

for reader in readers[1:]:
    for i,line in enumerate(reader):
        if 'acc' in line:
            break
        tag= line.strip().split('\t')[1]
        preds[i].append(tag)

with open('eng.dev.preds.txt','w') as writer:
    for idx, pred in preds.items():
        try:
            while True:
                pred.remove('NA')
        except ValueError:
            pass
        full_pred= pred[0] + '\t' +pred[1] + '\t'+ ';'.join(pred[2:]) + '\n'
        writer.write(full_pred)
    