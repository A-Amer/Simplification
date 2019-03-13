from torch import nn
from  torch import optim
from torch.nn.utils import clip_grad_norm_
import torch
import os
import random
import config
import seq2seqModel as s2s
import vocab
import glob


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
    simpleFiles = glob.glob(config.simpleTrainingFilePath)
    simpleFiles.sort()
    orgFiles=glob.glob(config.orgTrainingFilePath)
    orgFiles.sort()
    model, optimizer, startEpoch=loadCheckpoint(model, optimizer)
    numOfFiles=len(orgFiles)
#######################Training####################33
    for epoch in range(startEpoch,config.maxEpoch):
        model.train()
        fileIndecies=list(range(numOfFiles))
        random.shuffle(fileIndecies)
        for fileIndex in fileIndecies:
            with open(orgFiles[fileIndex]) as orgTxt:
                org=orgTxt.read().splitlines()
            orgTxt.close()
            with open(simpleFiles[fileIndex]) as simpleTxt:
                simple=simpleTxt.read().splitlines()
            simpleTxt.close()

            sentenceIndecies=list(range(len(org)))
            random.shuffle(sentenceIndecies)
            for sentenceIndex in sentenceIndecies:
                input=vocab.sentenceToTensor(org[sentenceIndex], orgVocab)
                target=vocab.sentenceToTensor(simple[sentenceIndex], simpleVocab)
                optimizer.zero_grad()
                output=model(input,len(target)+1)
                loss=lossFn(output[1:],target.squeeze(1))
                loss.backward()
                clip_grad_norm_(model.parameters(),config.gradientMax)
                optimizer.step()
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





train()
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









