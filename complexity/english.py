import helpers
import nltk
from collections import Counter
from collections import defaultdict
from itertools import groupby
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


featuresCount = 17
stemmer = PorterStemmer()
parser = CoreNLPParser('http://localhost:9005')


##Type-Token Ratio (TTR) is the ratio of number of word types (T) to total number word tokens in a text (N).
def lexicalDensity(tagged):
    taggedDict = defaultdict(int)
    totalWords = 0
    for word, tag in tagged:
        totalWords += 1
        type = helpers.getType(tag)
        taggedDict[type] += 1

    #taggedDict.pop("PUNCT")
    return taggedDict,totalWords

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

    features=np.empty(featuresCount)
    with open(filename, 'r') as f:
        text = f.read()
        ## Split document to sentences
        # sentences_list=re.split(r':|ِ،|\.|\؟|!|؛|،|\n|\t' ,txt)
        sentences = re.split(r'\.|\?|!|؛|\n|\t', text)
        #sentences=text.split(".")
        sentences.pop() ## remove last empty string

        for sent in sentences:

            if len(sent.split())>2 :
                # lowercase , remove punctuation
                sentence = sent.lower()
                sentence = sentence.translate(str.maketrans('', '', string.punctuation))
                tokens = nltk.word_tokenize(sentence)
                types = set()
                for token in tokens:
                    types.add(stemmer.stem(token))

                ##print(tokens)
                ##print("types", types)


                ## Feature: Type-Token Ratio
                t = len(types)
                n = len(tokens)
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
                else:  print("The same ",sent)

                ## Feature: Measure of Textual Lexical Diversity
                mtld = MTLD(tokens)
                ##print(mtld)

                ##Feature: lexical density
                tagged = nltk.pos_tag(tokens)
                taggedDict, totalWords = lexicalDensity(tagged)
                lexicalDens=0
                if totalWords!=0:
                    lexicalDens = len(taggedDict) / totalWords

                ## Verb Variation
                verbVariation=0

                ## Noun Variation
                nounVariation = 0

                ##Feature: clauses

                # nlp = StanfordCoreNLP('http://localhost:9005')
                # output= nlp.annotate(sent, properties={
                #   'annotators': 'tokenize,ssplit,pos,depparse,parse',
                #   'outputFormat': 'json'
                #   })
                ###print(output['sentences'][0]['parse'])


                sent=sent+"."
                ##print(sent)
                tree = list((parser.parse(parser.tokenize(sent))))
                #print(parseString)

                lengthSentence, countSentence, lengthClause, countClause = 0, 0, 0, 0
                countPP, countNP, countVP = 0, 0, 0
                lengthVP = 0
                countCoordinatingConj = 0
                countDependentClause = 0

                #tree = Tree.fromstring(string(parseString))
                # tree[0].draw()
                lengthTUnit = n
                ##print(lengthTUnit)
                for s in tree[0].subtrees():

                          if s.label()=="S":  ##Sentence
                            ##print("--------------------------------")
                            ##print(s.flatten())
                            ##print(len(s.flatten()))
                            countSentence+=1
                            lengthSentence+=len(s.flatten())
                          if helpers.isClause(s.label()):
                              ##print("+++++++++++++++CLAUSE++++++++++++++++")
                              ##print(s.flatten())
                              ##print(len(s.flatten()))
                              countClause+=1
                              lengthClause+=len(s.flatten())
                          if s.label() == "VP":
                              ##print("******************************")
                              ##print(s.flatten())
                              ##print(len(s.flatten()))
                              countVP += 1
                              lengthVP += len(s.flatten())
                          if s.label()=="NP":
                              ##print("NP NP NP NP NP NP NP NP")
                              if len(s.flatten())>2:
                                countNP += 1
                                ##print("Leaves ", s.leaves())
                          if s.label() == "PP":
                              ##print("PP PP PP PP PP PP PP PP")
                              countPP += 1
                              ##print(s)
                          if s.label()== "CC":
                              ##print("vvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvv")
                              ##print(s.flatten())
                              ##print(len(s.flatten()))
                              countCoordinatingConj+=1 ##check whether it is cc phrase or clause

                          if s.label()== "SBAR":
                              ##print("vvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvv")
                              ##print(s.flatten())
                              ##print(len(s.flatten()))
                              countDependentClause+=1 ##check whether it prints the actual length

                ##Todo : divide by zero exceptions
                meanLengthClause, meanLengthTUnit, meanLengthSentence, nClausesTUnit = 0, n, 0, countClause
                dependentClauseToClause, coordinatePhraseClause, coordinatePhraseTUnit = 0, 0, 0
                complexNominalsClause, complexNominalsTUnit = 0, countNP
                verbPhrasesTUnit, lengthVP = countVP, lengthVP

                if countClause > 0:
                    meanLengthClause = lengthClause / countClause
                    dependentClauseToClause = countDependentClause/countClause

                features = np.vstack((features,
                        [rootTTR, logTTR, uberIndexTTR, mtld , lexicalDens , nounVariation, verbVariation, meanLengthClause, meanLengthTUnit, nClausesTUnit,
                        dependentClauseToClause, coordinatePhraseClause, coordinatePhraseTUnit, complexNominalsClause, complexNominalsTUnit, verbPhrasesTUnit, lengthVP]))
    return features



def extractFeatures(folderPath):

    features = np.empty(featuresCount)
    for file in glob.glob(folderPath):
            features = np.vstack((features, featuresFromFile(file)))
    return features