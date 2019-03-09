from torch import nn
from  torch import optim
from torch.nn.utils import clip_grad_norm_
import torch
import os
import random
import config
import re
import seq2seqModel as s2s
import Vocab
import glob

class seq2Seq:
    def __init__(self):
        return
#try if validation does not increase stop for overfitting
def train(vocabFile=False):
    if vocabFile==False:
        Vocab.buildVocab(config.orgTrainingFilePath, config.orgVocabFilePath)
        Vocab.buildVocab(config.simpleTrainingFilePath, config.simpleVocabFilePath)
    #simpleVocab=Vocab.Vocab(config.simpleVocabFilePath)
    orgVocab=Vocab.Vocab(config.orgVocabFilePath)
    model=s2s.seq2seq(orgVocab.word2Id(Vocab.sentenceEnd),orgVocab.word2Id(Vocab.sentenceStart))
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
                target=Vocab.sentenceToTensor(Vocab.sentenceStart+simple[sentenceIndex]+Vocab.sentenceEnd,orgVocab)
                optimizer.zero_grad()
                output=model(input,len(target))
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

def chunk(fromPathOrg,fromPathSimple,ToPathOrg,ToPathSimple):
    i=0
    srcFileOrg=open(fromPathOrg)
    dstFileOrg=open(ToPathOrg+'/train_'+str(i)+'.src.txt',mode='w')
    srcFileSimple=open(fromPathSimple)
    dstFileSimple = open(ToPathSimple + '/train_' +str(i)+ '.dst.txt', mode='w')
    i+=1
    legalChars=r'[ ^\x00 -\x7Fé]+$'
    numOfExamples=0
    while True:
        lineOrg = srcFileOrg.readline().replace('-LCB-','{').replace('-RCB-','}').replace('-LRB-','(').replace('-RRB-',')').replace('-LSB-','[').replace('-RSB-',']').replace("â ''",'-')
        lineSimple = srcFileSimple.readline().replace('-LCB-','{').replace('-RCB-','}').replace('-LRB-','(').replace('-RRB-',')').replace('-LSB-','[').replace('-RSB-',']').replace("â ''",'-')
        if lineOrg=="":
            break
        if (not re.match(legalChars,lineOrg))  \
                or lineOrg[0].isnumeric() or lineSimple[0].isnumeric() or\
                not(( not lineSimple.endswith((".\n",";\n","!\n","?\n"))) or (lineOrg.endswith((".\n",";\n","!\n","?\n")))):
            continue
        dstFileOrg.write(lineOrg)
        dstFileSimple.write(lineSimple)
        numOfExamples+=1
        if numOfExamples==1000:
            dstFileOrg.close()
            dstFileSimple.close()
            dstFileOrg = open(ToPathOrg+'/train_'+str(i)+'.src.txt',mode='w')
            dstFileSimple = open(ToPathSimple + '/train_' + str(i) + '.dst.txt', mode='w')
            i+=1
            numOfExamples=0
    dstFileOrg.close()
    dstFileSimple.close()




#Vocab.buildVocab(config.orgTrainingFilePath,config.orgVocabFilePath)
Vocab.buildVocab(config.simpleTrainingFilePath, config.simpleVocabFilePath)











