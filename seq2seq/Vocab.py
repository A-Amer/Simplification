import glob
from nltk import word_tokenize
import torch

sentenceStart = '<s>'
sentenceEnd = '</s>'

padToken = '[PAD]' # This has a vocab id, which is used to pad the encoder input, decoder input and target sequence
unknownToken = '[UNK]' # This has a vocab id, which is used to represent out-of-vocabulary words
startDecoding = '[START]' # This has a vocab id, which is used at the start of every decoder input sequence
stopDecoding = '[STOP]' # This has a vocab id, which is used at the end of untruncated target sequences

class Vocab(object):

    def __init__(self,vocabFile,maxSize=0):
        self.word2Id={} #dictionary to transform word to id
        self.id2Word={} #dictionary to transform id to word
        self.count=0

        for w in [sentenceStart, sentenceEnd , padToken , unknownToken, startDecoding, stopDecoding]:
            self.word2Id[w]=self.count
            self.id2Word[self.count]=w
            self.count+=1

        with open(vocabFile) as vocab:
            for w in vocab:
                w=w.replace("\n","")
                self.word2Id[w] = self.count
                self.id2Word[self.count] = w
                self.count += 1
                if self.count==maxSize and maxSize!=0:
                    vocab.close()
                    break

    def wordToId(self,word):
        if word not in self.word2Id:
            return self.word2Id[unknownToken]
        return self.word2Id[word]

    def idToWord(self,id):
        if id not in self.id2Word:
            raise  ValueError("id not in vocab")
        return self.id2Word[id]
    def size(self):
        return self.count

def sentenceToIds(sentence,vocab):
    return [vocab.wordToId(word) for word in sentence.split(' ')]

def sentenceToTensor(sentence,vocab):
    indexes = sentenceToIds(sentence,vocab)
    indexes.append(vocab.wordToId(sentenceEnd))
    return torch.tensor(indexes, dtype=torch.long).view(-1, 1)
def batchToPaddedSeq(batch,vocab):
    sequences=[]
    for sentence in batch:
        sequences.append(sentenceToTensor(sentence,vocab))

    return torch.nn.utils.rnn.pack_sequence(sequences,padding_value=vocab.wordToId(padToken))

def articleToId(articleTokens,vocab):
    oov=[]
    ids=[]
    oovId=0
    unk_id = vocab.word2id(unknownToken)
    for word in articleTokens:
        id=vocab.wordToId(word)
        if id==unk_id:#if out of vocab word
            if word not in oov:
                oov.append(word)

            oovIndex=oov.index(word)
            ids.append(vocab.size()+oovIndex)#The id of the out of contex word is the size+id of oov word

        else:#word not out of vocab so add its regular id
            ids.append(id)

    return ids,oov

def abstractToId(abstractTokens,vocab,articleOov):
    ids = []
    oovId = 0
    unk_id = vocab.word2id(unknownToken)
    for word in abstractTokens:
        id = vocab.wordToId(word)
        if id==unk_id:
            if word in articleOov:#Token is an OOV found in the original article
                oovIndex = articleOov.index(word)
                ids.append(vocab.size() + oovIndex)
            else:#The token is UNK and not present in the article
                ids.append(unk_id)

        else:
            ids.append(id)
    return ids

def outputIdToWords(ids,vocab,articleOov):
    wordList=[]
    for id in ids:
        try:
            word=vocab.id2Word(word)
        except ValueError as e:
            assert articleOov is not None, "Error: model produced a word ID that isn't in the vocabulary. This should not happen in baseline (no pointer-generator) mode"
            oovId=id-vocab.size()#if id has no word it could be an OOV
            try:
                word=articleOov[oovId]
            except ValueError as e:
                raise ValueError("Error model produced an id that does not exsist")

        wordList.append(word)

    return  wordList
########################################
def buildVocab(path,vocabFilePath):#path:regex of the path+format of files to extract the vocab eg:'/home/download/*.txt'
    files=glob.glob(path)
    wordCount={}
    for file in files:
        try:
            with open(file) as f:
                txt=f.read()
                textTokens=word_tokenize(txt)
                for word in textTokens:
                    if word not in wordCount:
                        wordCount[word]=0
                    wordCount[word]+=1
                f.close()
        except IOError as exc:
                print( exc)

    vocabFile=open(vocabFilePath,'w')
    for key, value in sorted(wordCount.items(), key=lambda kv: kv[1], reverse=True):
        if value>=2:
            vocabFile.write(key+'\n')
        else:
            break

    vocabFile.close()
