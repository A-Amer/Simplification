from torch import nn
from  torch import optim
from torch.nn.utils import clip_grad_norm_

import torch
import os
import random
import config

import Vocab
import glob

class seq2Seq:
    def __init__(self):
        return
#try if validation does not increase stop for overfitting
def train(vocabFile=False):
    if vocabFile==False:
        Vocab.buildVocab(config.fullOrgTxtPath,config.orgVocabFilePath)
        Vocab.buildVocab(config.fullSimpleTxtPath, config.simpleVocabFilePath)
    #simpleVocab=Vocab.Vocab(config.simpleVocabFilePath)
    orgVocab=Vocab.Vocab(config.orgVocabFilePath)
    model=seq2Seq()
    lossFn=nn.NLLLoss()
    optimizer=optim.Adam(model.parameters())
    simpleFiles = glob.glob(config.simpleTrainingFilePath)
    orgFiles=glob.glob(config.orgTrainingFilePath)
    model, optimizer, startEpoch=loadCheckpoint(model, optimizer)
    numOfFiles=len(orgFiles)

    for epoch in range(startEpoch,config.maxEpoch):
        model.train()
        fileIndecies=range(numOfFiles)
        random.shuffle(fileIndecies)
        for fileIndex in fileIndecies:
            with open(orgFiles[fileIndex]) as orgTxt:
                org=orgTxt.read().splitlines()
            orgTxt.close()
            with open(simpleFiles[fileIndex]) as simpleTxt:
                simple=simpleTxt.read().splitlines()
            simpleTxt.close()

            sentenceIndecies=range(len(org))
            random.shuffle(sentenceIndecies)
            for sentenceIndex in sentenceIndecies:
                input=Vocab.sentenceToTensor(org[sentenceIndex]+Vocab.sentenceEnd,orgVocab)
                target=Vocab.sentenceToTensor(simple[sentenceIndex]+Vocab.sentenceEnd,orgVocab)
                optimizer.zero_grad()
                output=model(input,len(simple[sentenceIndex]))
                loss=lossFn(output,target)
                loss.backward()
                clip_grad_norm_(model.paramters(),config.gradientMax)
                optimizer.step()
        states={'epoch': epoch + 1, 'stateDict': model.state_dict(),
             'optimizer': optimizer.state_dict(),  }
        torch.save(states, config.checkpointPath)


def eval(model,simpleDataPath,orgDataPath):
    # simpleVocab=Vocab.Vocab(config.simpleVocabFilePath)
    model.eval()
    orgVocab = Vocab.Vocab(config.orgVocabFilePath)
    lossFn = nn.NLLLoss()
    simpleFiles = glob.glob(simpleDataPath)
    orgFiles = glob.glob(orgDataPath)
    lossTotal=0
    numOfEx=0
    fileIndecies = range(len(orgFiles))
    for fileIndex in fileIndecies:
        loss=0
        with open(orgFiles[fileIndex]) as orgTxt:
            org = orgTxt.read().splitlines()
        orgTxt.close()
        with open(simpleFiles[fileIndex]) as simpleTxt:
            simple = simpleTxt.read().splitlines()
        simpleTxt.close()
        sentenceIndecies = range(len(org))
        numOfEx+=len(org)
        for sentenceIndex in sentenceIndecies:
            input = Vocab.sentenceToTensor(org[sentenceIndex] + Vocab.sentenceEnd, orgVocab)
            target = Vocab.sentenceToTensor(simple[sentenceIndex] + Vocab.sentenceEnd, orgVocab)
            output = model(input, len(simple[sentenceIndex]))
            loss += lossFn(output, target)
        print('File {} has avarage loss of {:.4f}'.format(fileIndex+1,loss/len(org)))
        lossTotal+=loss
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
def chunk(fromPath,ToPath):
    i=0
    srcFile=open(fromPath)
    dstFile=open(ToPath+'/train_'+i+'.bin',mode='w')
    i+=1
    line=srcFile.readline()
    numOfExamples=0
    while line!="":
        dstFile.write(line)
        numOfExamples+=1
        if numOfExamples==1000:
            dstFile.close()
            dstFile = open(ToPath + '/train_' + i + '.bin', mode='w')
            i+=1
            numOfExamples=0
        line = srcFile.readline()




vocab=torch.load("/home/amira/GP-imp/Abstractive Sum/data-simplification/wikilarge/wiki.full.aner.train.dst.vocab.tmp.t7")
print("hello")










