import config
from torch import topk
from torch import log

class searchNode(object):
    def __init__(self,parentSeq,score,prob):
        self.seq=parentSeq
        self.p=prob
        self.score=score

def scores(element):
    return element.score

def beamSearchDecoder(maxLen,intialInput,decoder, encoderStates, hidden, cell,eos):
    hypothesis=[]
    candidate=[]
    finalList=[]
    krange=range(config.beamSize)
    output, hidden, cell = decoder(intialInput, encoderStates, hidden, cell)
    val, ind = topk(log(output), k=config.beamSize)
    for i in krange:
        hypothesis.append(searchNode([ind[0][i].unsqueeze(0)],val.squeeze(0)[i].data,[output]))

    while len(hypothesis)!=0 :
        for i in range(len(hypothesis)):
            node=hypothesis[i]
            input = node.seq[-1]
            output, hidden, cell = decoder(input, encoderStates, hidden, cell)
            val, ind = topk(log(output), k=config.beamSize)
            for j in krange:
                seq = node.seq.copy()
                seq.append(ind[0][j].unsqueeze(0))
                candidate.append(searchNode(seq,node.score+val.squeeze(0)[j].data))
        hypothesis.clear()
        candidate.sort(key=scores,reverse=True)
        for sol in candidate[0:config.beamSize]:
            if len(sol.seq)>=maxLen or sol.seq[-1].squeeze(0)==eos:
                finalList.append(sol)
            else:
                hypothesis.append(sol)
        candidate.clear()
        if len(finalList)>=config.beamSize:
            break
    finalList.sort(key=scores,reverse=True)
    return  finalList[0].seq

