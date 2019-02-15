import  Preprocess
import PSO
import numpy as np


def informativeScoreCalc(idf,maxSentenceLen,noOfSentences,tokenizedSentences,sentenceMaps,wordMap):
    infScore=np.zeros(noOfSentences)
    tf_idf=np.zeros((noOfSentences,len(list(idf.values()))))
    for key in list(idf.keys()):
        idf[key]=np.log(int(noOfSentences)/idf[key])
    for i in range (noOfSentences):
        infScore[i]=(len(tokenizedSentences[i])/maxSentenceLen)+(1-(i+1)/noOfSentences)
        tf=dict.fromkeys(idf,0)
        for key in list(sentenceMaps[i].keys()):
            tf[key]=sentenceMaps[i][key]/wordMap[key]

        tf_idf[i]=np.multiply((list(tf.values())),(list(idf.values())))
        infScore[i]+=np.sum(tf_idf[i])
        tf.clear()
    sentenceMaps.clear()
    idf.clear()
    tokenizedSentences.clear()
    wordMap.clear()
    return infScore,tf_idf
def similarityMatrix(noOfSentences,tf_idf):
    dag = np.zeros((noOfSentences, noOfSentences))
    for i in range(noOfSentences-1):
        for j in range(i+1,noOfSentences):
            dag[i][j]=np.matmul(tf_idf[i],tf_idf[j])/(np.linalg.norm(tf_idf[i])*np.linalg.norm(tf_idf[j]))
    #tf_idf.clear()
    return dag

def intialize():
    sentences,textLen,sentencesMaps, maxSentenceLen, wordMap, tokenizedSentences,idf= Preprocess.textPreprocessing("Untitled Document.txt")
    infScore, tf_idf=informativeScoreCalc(idf, maxSentenceLen, len(sentences), tokenizedSentences, sentencesMaps, wordMap)
    dag=similarityMatrix(len(sentences),tf_idf)
    return infScore,dag,sentences,textLen

infScore,simMatrix,sentences,textLen=intialize()
psoObject=PSO.PSO(sentences,simMatrix,infScore,0.7*textLen,20,700,0)
sol=psoObject.getBestSolution()
indx=list(np.where(sol==1)[0])
for i in indx:
    print(sentences[i])

