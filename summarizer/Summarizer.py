import Preprocess
import PSO
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

def informativeScoreCalc(idf,maxSentenceLen,noOfSentences,tokenizedSentences,sentenceMaps,n):
    infScore=np.zeros(noOfSentences)
    tf_idf=np.zeros((noOfSentences,len(list(idf.values()))))

    for i in range (noOfSentences):
        infScore[i]=(len(tokenizedSentences[i])/maxSentenceLen)+(1-(i+1)/noOfSentences)
        tf=dict.fromkeys(idf,0)
        for key in list(sentenceMaps[i].keys()):
            tf[key]=sentenceMaps[i][key]/n[key]

        tf_idf[i]=np.multiply((list(tf.values())),(list(idf.values())))
        infScore[i]+=np.sum(tf_idf[i])
        tf.clear()
    sentenceMaps.clear()
    idf.clear()
    tokenizedSentences.clear()
    n.clear()
    return infScore


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
        #print(sentences[i])
        extractiveSum+=sentences[i]
    #print("\n PSO\n")
    infScore=informativeScoreCalc(idf, avgDL, len(sentences), tokenizedSentences, sentencesMaps, n)
    return infScore,dag,sentences,textLen,extractiveSum

orgPath="News Articles/business/001.txt"
summaryPath="Summaries/business/001.txt"
infScore,simMatrix,sentences,textLen,exSum=intialize(0.35,orgPath)
psoObject=PSO.PSO(sentences,simMatrix,infScore,0.4*textLen,20,700,0)
sol=psoObject.getBestSolution()
indx=list(np.where(sol==1)[0])
psoSum=""
for i in indx:
    #print(sentences[i])
    psoSum+=sentences[i]
ref=open(summaryPath).read()
r=rouge.Rouge()
print(r.get_scores(exSum,ref))
print(r.get_scores(psoSum,ref))

