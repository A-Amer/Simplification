import helpers
import nltk
from collections import defaultdict
import math
import string
import re
from pycorenlp import StanfordCoreNLP
from nltk.corpus import treebank
from nltk import Tree
from nltk.draw.tree import draw_trees
from nltk.parse import CoreNLPParser
from nltk.stem import PorterStemmer
import numpy as np
import glob
import pyphen

featuresCount = 20
stemmer = PorterStemmer()
parser = CoreNLPParser('http://localhost:9005')
academicWordsList =  set(open('academic-word-list.txt').read().split()) ## academic word list feature

##Type-Token Ratio (TTR) is the ratio of number of word types (T) to total number word tokens in a text (N).
def lexicalDensity(tagged):
    taggedDict = defaultdict(int)
    modifierVariation , adverbVariation= 0 ,0
    nounVariation , adjectiveVariation=0,0
    totalwords=0

    for word, tag in tagged:
        totalwords+=1
        type = helpers.getType(tag)
        taggedDict[type] += 1
        if type == "ADV": adverbVariation += 1
        if type == "PRN": modifierVariation += 1
        if type == "NN": nounVariation += 1
        if type == "ADJ": adjectiveVariation += 1

    return len(taggedDict)/totalwords,adverbVariation/totalwords , modifierVariation/totalwords , nounVariation/totalwords , adjectiveVariation/totalwords

## Measure of Textual Lexical Diversity
def MTLD(tokens):
    mtldCount=0
    defaultTTR=0.72
    totalWords= 0#len(tagged)
    types=[]
    for t in tokens:
        totalWords+=1
        if type not in types:
            types.append(type)
        ttr=len(types)/totalWords
        if ttr<= defaultTTR :
            mtldCount+=1
            totalWords=0
            types=[]
    return mtldCount



def featuresFromFile(filename):

    features=np.zeros([0,featuresCount])
    with open(filename, 'r') as f:
        text = f.read()
        ## Split document to sentences
        sentences = re.split(r'\n', text)
        sentences.pop() ## remove last empty string

        for sent in sentences:


            if len(sent.split())>2 :
                # lowercase , remove punctuation

                sentence = sent.lower()
                sentence = sentence.translate(str.maketrans('', '', string.punctuation)) ## remove punctuation
                #print(sentence[0:4])
                tokens = nltk.word_tokenize(sentence)
                types = set()
                numSyllables=0
                awlRatio=0
                letters=0

                dic = pyphen.Pyphen(lang='nl_NL')
                for token in tokens:
                    letters+=len(token)
                    numSyllables += dic.inserted(token).count('-') + 1
                    types.add(stemmer.stem(token))
                    if token in academicWordsList: awlRatio+=1

                ## Feature: Type-Token Ratio
                t = len(types)
                n = len(tokens)
                numSyllables /= n
                awlRatio/=n
                lettersToWords=letters/n


                ttr , logTTR, rootTTR =0,0,0

                if n!=0:
                    ttr = t / n
                    rootTTR = t / math.sqrt(n)
                    if math.log(n, 2) != 0:
                        logTTR = math.log(t, 2) / math.log(n, 2)
                    else:
                        print("Log =0 ", sent)

                else : print("N=0", sent)




                uberIndexTTR = 0

                if n!=t:
                    uberIndexTTR = math.pow(math.log(t, 2), 2) / math.log(n / t)


                ## Feature: Measure of Textual Lexical Diversity
                mtld = MTLD(tokens)
                ##print(mtld)

                ##Feature: lexical density
                tagged = nltk.pos_tag(tokens)
                lexicalDens, adverbVariation,modifierVariation,nounVariation,adjectiveVariation = lexicalDensity(tagged)

                sent=sent+"."

                tree = list((parser.parse(parser.tokenize(sent))))
                #print(parseString)

                lengthSentence, countSentence, lengthClause, countClause = 0, 0, 0, 0
                countPP, countNP, countVP = 0, 0, 0
                lengthVP = 0
                countCoordinatingConj = 0
                countDependentClause = 0

                lengthTUnit = n

                for s in tree[0].subtrees():

                          if s.label()=="S":  ##Sentence
                            countSentence+=1
                            lengthSentence+=len(s.flatten())
                          if helpers.isClause(s.label()):
                              countClause+=1
                              lengthClause+=len(s.flatten())
                          if s.label() == "VP":
                              countVP += 1
                              lengthVP += len(s.flatten())
                          if s.label()=="NP":
                              if len(s.flatten())>2:
                                countNP += 1

                          if s.label() == "PP":
                              countPP += 1

                          if s.label()== "CC":
                              countCoordinatingConj+=1 ##TO DO: check whether it is cc phrase or clause

                          if s.label()== "SBAR":
                              countDependentClause+=1 ##check whether it prints the actual length

                ##Todo : divide by zero exceptions
                meanLengthClause, meanLengthTUnit, meanLengthSentence, nClausesTUnit = 0, n, 0, countClause
                dependentClauseToClause, coordinatePhraseClause, coordinatePhraseTUnit = 0, 0, 0
                complexNominalsTUnit = countNP
                verbPhrasesTUnit, lengthVP = countVP, lengthVP

                if countClause > 0:
                    meanLengthClause = lengthClause / countClause
                    dependentClauseToClause = countDependentClause/countClause
                if countVP>1:
                    lengthVP=lengthVP/countVP
                if countDependentClause>1:
                    colemanLiau = letters / n * 100 * 0.0588 - countDependentClause / n * 100 * 0.296 - 15.8
                else: colemanLiau = letters / n * 100 * 0.0588 - 1 / n * 100 * 0.296 - 15.8

                print(sent[0:6])
              #  print("Coleman Liau", colemanLiau)
             #   print("Average Syllables", numSyllables)

                sentFeatures= np.array([colemanLiau,  logTTR, uberIndexTTR, rootTTR, mtld, awlRatio, numSyllables,
                         lexicalDens, adverbVariation,modifierVariation,adjectiveVariation, nounVariation,
                         meanLengthClause, meanLengthTUnit, nClausesTUnit,
                        dependentClauseToClause, complexNominalsTUnit, verbPhrasesTUnit , lengthVP , countCoordinatingConj])
                features = np.vstack((features,sentFeatures))

    return features



def extractFeatures(folderPath):

    features = np.empty(featuresCount)
    for file in glob.glob(folderPath):
            features = np.vstack((features, featuresFromFile(file)))
    return features