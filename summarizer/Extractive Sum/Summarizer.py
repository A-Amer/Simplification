import Preprocess
import numpy as np
import rouge

def updateIdf(n,noOfSentences,eps=0.25):
    avgIdf=0
    idf = {}
    for key in list(n.keys()):
        idf[key]=np.log(int(noOfSentences)/n[key])
        avgIdf+=idf[key]
    idfKeys=list(idf.keys())
    avgIdf=avgIdf/len(idfKeys)
    for key in idfKeys:
        if (noOfSentences-n[key]+0.5)<=0:
            continue
        if n[key]>noOfSentences/3:
            idf[key]=np.log((noOfSentences-n[key]+0.5)/(n[key]+0.5))
        else:
            idf[key]=eps*avgIdf
    return idf

def getScore(index1,index2,idf,avgDL,sentencesMaps,tokenizedSentences,k=1.2,b=0.75):
    score=0
    dl=len(tokenizedSentences[index2])
    for word in tokenizedSentences[index1]:
        if word not in sentencesMaps[index2]:
            continue
        den=1-b+b*(dl/avgDL)
        den=den*k
        den+=sentencesMaps[index2][word]
        score+=(idf[word]*sentencesMaps[index2][word]*(k+1))/den
    return score

def createGraph(n,noOfSentences,avgDL,sentencesMaps,tokenizedSentences):
    idf=updateIdf(n,noOfSentences)
    dag = np.zeros((noOfSentences, noOfSentences))
    for i in range(noOfSentences-1):
        for j in range(i+1,noOfSentences):
            dag[i][j] =getScore(i,j,idf,avgDL,sentencesMaps,tokenizedSentences)
    return dag,idf

def pagerank_weighted(graph,noOfSentences, initial_value=None, damping=0.85):
    """Calculates PageRank for an undirected graph"""
    convergenceThreshold = 0.0001
    if initial_value == None: initial_value = 1.0 / noOfSentences
    scores = dict.fromkeys(range(noOfSentences), initial_value)

    iteration_quantity = 0
    for iteration_number in range(100):
        iteration_quantity += 1
        convergence_achieved = 0
        for i in range(noOfSentences):
            rank = 1 - damping
            for j in range(i + 1, noOfSentences):
                neighbors_sum=sum(graph[j])
                if neighbors_sum==0:
                    continue
                rank += damping * scores[j] * graph[i][j] / neighbors_sum
            if abs(scores[i] - rank) <= convergenceThreshold:
                convergence_achieved += 1

            scores[i] = rank

        if convergence_achieved == noOfSentences:
            break

    return scores


def intialize(ratio,txtPath):
    sentences, textLen, sentencesMaps, avgDL, n, tokenizedSentences= Preprocess.textPreprocessing(txtPath)
    dag,idf=createGraph(n,len(sentences),avgDL,sentencesMaps,tokenizedSentences)
    scores=pagerank_weighted(dag,len(sentences))
    ratio=ratio*len(sentences)
    sumSentences=[]
    i=0
    for key, value in sorted(scores.items(), key=lambda kv: kv[1],reverse=True):
        sumSentences.append(key)
        i+=1
        if i>ratio:
            break
    sumSentences.sort()
    extractiveSum=""
    for i in sumSentences:
        extractiveSum+=sentences[i]

    return dag,sentences,textLen,extractiveSum

orgPath="News Articles/business/002.txt"
summaryPath="Summaries/business/002.txt"
simMatrix,sentences,textLen,exSum=intialize(0.4,orgPath)

ref=open(summaryPath).read()
r=rouge.Rouge()
print(r.get_scores(exSum,ref))

