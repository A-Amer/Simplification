import io
import re
import numpy as np
from nltk.stem.isri import ISRIStemmer
from nltk.tokenize import wordpunct_tokenize
from nltk.corpus import stopwords


def readFile(fileName):
    openedFile=io.open(fileName)
    text=openedFile.read()
    openedFile.close()
    return text
def segment(text):
    sentences=re.split(r'[!;:\.\?]',str(text))
    return sentences,len(text)

def stemAndStopWords(text,wordMap,st,stops,idf):
    sentenceWordMap={}
    word_tokens = wordpunct_tokenize(text)
    filteredSentence = []
    for w in word_tokens:
        stem=st.stem(w)
        if stem not in stops:
            filteredSentence.append(stem)
            if stem in sentenceWordMap:
                sentenceWordMap[stem]+=1
            else:
                sentenceWordMap[stem] = 1
                if stem in idf:
                    idf[stem] += 1
                else:
                    idf[stem] = 1

            if stem in wordMap:
                     wordMap[stem]+=1
            else:
                     wordMap[stem] = 1
    return filteredSentence,sentenceWordMap



def tokenize(txtList):
    maxSentenceLen=0
    stops = set(stopwords.words('arabic'))
    st = ISRIStemmer()
    wordMap={}
    idf={}
    sentencesMaps=[]
    tokenizedSentences=[]
    i=0
    while i< len(txtList):
        sentence=str(txtList[i]).replace('.',' ').replace('!',' ').replace('?',' ') \
            .replace('\n',' ').replace('؟',' ').replace('،',' ').replace('"',' ')
        sentence=' '.join(sentence.split())
        filteredSentence, sentenceWordMap=stemAndStopWords(sentence, wordMap, st, stops,idf)
        if len(filteredSentence)>maxSentenceLen:
            maxSentenceLen=len(filteredSentence)
        elif len(filteredSentence)<=1:
            txtList.pop(i)
            continue
        tokenizedSentences.append(filteredSentence)
        sentencesMaps.append(sentenceWordMap)
        i+=1
    return sentencesMaps,maxSentenceLen,wordMap,tokenizedSentences,idf

def textPreprocessing(file):
    sentences,textLen=segment(readFile(file))
    sentencesMaps, maxSentenceLen, wordMap, tokenizedSentences, idf=tokenize(sentences)
    return sentences,textLen,sentencesMaps, maxSentenceLen, wordMap, tokenizedSentences, idf

#textPreprocessing("Untitled Document.txt")


