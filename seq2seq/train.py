from torch import nn
from torch import optim
from torch.nn.utils import clip_grad_norm_
import torch
import os
import random
import config
import seq2seqModel as s2s
import vocab
import glob
import time

#try if validation does not increase stop for overfitting
def train():
    ###########inialize model and optimizer etc.#############
    simpleVocab=vocab.vocab(config.simpleVocabFilePath, maxSize=config.vocabSizeDec)
    orgVocab=vocab.vocab(config.orgVocabFilePath, maxSize=config.vocabSizeEnc)

    config.vocabSizeEnc=orgVocab.size()
    config.vocabSizeDec=simpleVocab.size()

    model=s2s.seq2seq(orgVocab.wordToId(vocab.sentenceEnd), orgVocab.wordToId(vocab.sentenceStart))
    lossFn=nn.NLLLoss()
    optimizer=optim.Adam(model.parameters(),lr=config.learningRate,betas=(config.beta1,config.beta2))

    simpleFiles = glob.glob(config.simpleTrainingFilePath+'/*.txt')
    simpleFiles.sort()
    orgFiles=glob.glob(config.orgTrainingFilePath+'/*.txt')
    orgFiles.sort()

    model, optimizer, startEpoch=loadCheckpoint(model, optimizer)
    numOfFiles=len(orgFiles)
    pytorch_total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print('No. of trainable parameters: {}'.format(pytorch_total_params))
#######################Training######################
    for epoch in range(startEpoch,config.maxEpoch):
        print('start of epoch no.: {}'.format(epoch))
        model.train()
        fileIndecies=list(range(numOfFiles))
        random.shuffle(fileIndecies)
        fileIndecies[0]=76
        for fileIndex in [0]:
            print('start of file no.: {}'.format(fileIndex))
            with open(orgFiles[fileIndex]) as orgTxt:
                org=orgTxt.read().splitlines()
            orgTxt.close()
            with open(simpleFiles[fileIndex]) as simpleTxt:
                simple=simpleTxt.read().splitlines()
            simpleTxt.close()

            sentenceIndecies=list(range(len(org)))
            random.shuffle(sentenceIndecies)
            t = time.time()

            for i in range(config.batchPerFile):
                batch=list()
                actualSeqLen=list()
                target=list()
                for j in range(config.batchSize):
                    batch.append(vocab.sentenceToTensor(org[sentenceIndecies[100*i+j]], orgVocab))
                    actualSeqLen.append(len(org[sentenceIndecies[100*i+j]]))
                    target.append(vocab.sentenceToTensor(simple[sentenceIndecies[100*i+j]], simpleVocab))
                input=nn.utils.rnn.pad_sequence(batch, batch_first=False)
                actualSeqLen=torch.tensor(actualSeqLen, dtype=torch.long)

                optimizer.zero_grad()
                output=model(input,len(target)+1)
                loss=lossFn(output[1:],target.squeeze(1))
                loss.backward()
                clip_grad_norm_(model.parameters(),config.gradientMax)
                optimizer.step()
                loss.detach()
            now = time.time()
            print(now-t)
            #print(sentenceIndex)
        #save a checkpoint each epoch
        states={'epoch': epoch + 1, 'stateDict': model.state_dict(),
             'optimizer': optimizer.state_dict(),  }
        torch.save(states, config.checkpointPath)
        print("epoch number {} finished".format(epoch))


def evalModel(model,simpleVocab,orgVocab,simpleDataPath,orgDataPath):
    model.eval()
    lossFn = nn.NLLLoss()
    lossTotal=0
    numOfEx=0

    with open(orgDataPath) as orgTxt:
        org = orgTxt.read().splitlines()
    orgTxt.close()
    with open(simpleDataPath) as simpleTxt:
        simple = simpleTxt.read().splitlines()
    simpleTxt.close()
    sentenceIndecies = range(len(org))
    numOfEx+=len(org)
    for sentenceIndex in sentenceIndecies:
        input = vocab.sentenceToTensor(org[sentenceIndex] , orgVocab)
        target = vocab.sentenceToTensor(simple[sentenceIndex] , simpleVocab)
        output = model(input, len(target)+1)
        lossTotal += lossFn(output[1:], target.squeeze(1))

    print('Total avarage loss of {:.4f}'.format(lossTotal /numOfEx))

def loadCheckpoint(model, optimizer, filename=config.checkpointPath):
    # Note: Input model & optimizer should be pre-defined.  This routine only updates their states.
    startEpoch = 0
    if os.path.isfile(filename):
        checkpoint = torch.load(filename)
        startEpoch = checkpoint['epoch']
        model.load_state_dict(checkpoint['stateDict'])
        optimizer.load_state_dict(checkpoint['optimizer'])


    else:
        print("=> no checkpoint found at '{}'".format(filename))

    return model, optimizer, startEpoch



#print(torch.cuda.is_available())
train()
#trial 1: start 4:30
'''simpleVocab=vocab.vocab(config.simpleVocabFilePath, maxSize=config.vocabSizeDec)
orgVocab=vocab.vocab(config.orgVocabFilePath, maxSize=config.vocabSizeEnc)
config.vocabSizeEnc=orgVocab.size()
config.vocabSizeDec=simpleVocab.size()
model=s2s.seq2seq(orgVocab.wordToId(vocab.sentenceEnd), orgVocab.wordToId(vocab.sentenceStart))
model.eval()
with open('data/test_0.src.txt') as orgTxt:
    org = orgTxt.read().splitlines()
orgTxt.close()
with open('data/test_0.dst.txt') as simpleTxt:
    simple = simpleTxt.read().splitlines()
simpleTxt.close()
sentenceIndecies = range(len(org))

for sentenceIndex in sentenceIndecies:
    input = vocab.sentenceToTensor(org[sentenceIndex], orgVocab)
    target = vocab.sentenceToTensor(simple[sentenceIndex], simpleVocab)
    output = model.run(input,len(target)+1)'''





''' vocabsize=50000 ---> 1 epoch = 915.6837964057922 sec --> file no. = 76
    vocabsize=0--------> 1 epoch = 937.8973233699799 sec --> file no. =76
'''



