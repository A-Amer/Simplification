import re
import glob
from nltk import word_tokenize
import config

def chunk(fromPathOrg,fromPathSimple,ToPathOrg,ToPathSimple):
    i=0
    srcFileOrg=open(fromPathOrg)
    dstFileOrg=open(ToPathOrg+'/train_'+str(i)+'.src.txt',mode='w')
    srcFileSimple=open(fromPathSimple)
    dstFileSimple = open(ToPathSimple + '/train_' +str(i)+ '.dst.txt', mode='w')
    i+=1
    legalChars=r'[ ^\x00 -\x7Fé]+$'
    examplesPerFile = config.batchSize*config.batchPerFile
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
        dstFileOrg.write(lineOrg.lower())
        dstFileSimple.write(lineSimple.lower())
        numOfExamples+=1
        if numOfExamples==examplesPerFile:
            dstFileOrg.close()
            dstFileSimple.close()
            dstFileOrg = open(ToPathOrg+'/train_'+str(i)+'.src.txt',mode='w')
            dstFileSimple = open(ToPathSimple + '/train_' + str(i) + '.dst.txt', mode='w')
            i+=1
            numOfExamples=0
    dstFileOrg.close()
    dstFileSimple.close()

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


def preprocessOriginalData():
    chunk(config.fullOrgTxtPath, config.fullSimpleTxtPath, config.orgTrainingFilePath, config.simpleTrainingFilePath)
    buildVocab(config.orgTrainingFilePath, config.orgVocabFilePath)
    buildVocab(config.simpleTrainingFilePath, config.simpleVocabFilePath)

preprocessOriginalData()
