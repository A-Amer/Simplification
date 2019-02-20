import  nlpParser as stanford
import numpy as np
import Preprocess as pre
import io
import nltk.tree as tree


def matchWords(sourceStr,destStr):
    matched = []
    for i in range(len(destStr)):
        for j in range(len(sourceStr)):
            if destStr[i]==sourceStr[j]:
                matched.append((j,i))
                break
    return matched
def alignTrees(source,dest,matchedWords):
    completeMatch=[]
    parentCountD={}
    parentCountS={}
    largestPartial={}
    for pair in matchedWords:
        sourcePos=source.leaf_treeposition(pair[0])
        destPos=dest.leaf_treeposition(pair[1])
        a = len(sourcePos) - 1
        b = len(destPos) - 1
        while a >= 0 or b >= 0:
            if source[sourcePos[0:a]] == dest[destPos[0:b]]:
                a -= 1
                b -= 1
            else:
                completeMatch.append((sourcePos[0:a + 1], destPos[0:b + 1]))#if tree does not match then complete match at the previous subtree
                largestPartial[sourcePos[0:a]]=(sourcePos[0:a + 1], destPos[0:b + 1])
                if destPos[0:b] in parentCountD:
                    parentCountD[destPos[0:b]]+=1
                    x=b-1
                    while x>=0:
                        if destPos[0:x] in parentCountD:
                            parentCountD[destPos[0:x]] += 1
                            x-=1
                        else:
                            parentCountD[destPos[0:x]] = 1


                else:
                    parentCountD[destPos[0:b]]=1

                if sourcePos[0:a] in parentCountS:
                    parentCountS[sourcePos[0:a]] += 1
                    x = a - 1
                    while x >= 0:
                        if sourcePos[0:x] in parentCountS:
                            parentCountS[sourcePos[0:x]] += 1
                            x -= 1
                        else:
                            parentCountS[sourcePos[0:x]] = 1
                            break
                else:
                    parentCountS[sourcePos[0:a]] = 1

                break

    for pair in completeMatch:
        sourcePos=pair[0]
        destPos=pair[1]
        a = len(pair[0])-1
        b = len(pair[1])-1
        while a >= 0 and b >= 0:
            if source[sourcePos[0:a]].label() == dest[destPos[0:b]].label() \
                    and ((len(dest[destPos[0:b]])==1 and len(source[sourcePos[0:a]])==1)
                         or (parentCountD[destPos[0:b]]>=2 and parentCountS[sourcePos[0:a]]>=2) ):
                if sourcePos[0:a] in largestPartial:
                    largestPartial.pop(sourcePos[0:a])
                largestPartial[sourcePos[0:a-1]]=(sourcePos[0:a],destPos[0:b])
                a-=1
                b-=1
            else:
                break
    parentCountD.clear()
    parentCountS.clear()
    partialMatch=list(largestPartial.values())
    largestPartial.clear()
    return completeMatch,partialMatch


def extractRules(source,dest,completeMatch,partialMatch,rulesRhs,rulesLhs):

    for pair in partialMatch:
        sourceStr = str(source[pair[0]])
        if pair in completeMatch:
            continue
        sourceTupleLen=len(pair[0])
        destStr=str(dest[pair[1]])
        if pair[1]==():
            destStr=str(dest).replace('\n','=')
        for matches in completeMatch:
            sourceStr = sourceStr.replace(str(source[matches[0]]), '(' + source[matches[0]].label() + '.*?' + ')')
            if len(matches[0])>=sourceTupleLen and matches[0][0:sourceTupleLen]==pair[0]:
                newString='['+str(matches[0][sourceTupleLen:])+']'
                destStr=destStr.replace(str(dest[matches[1]]),newString)
                #dest[matches[1]].set_label(str((0,)+matches[0][sourceTupleLen:]))

        rulesRhs.append(destStr)
        if pair[0]==(0,):
            rulesLhs.append(str(sourceStr).replace('\n','='))
        else:
            rulesLhs.append(sourceStr)


def extractSplit(source,dest,partialMatches,rulesRhs,rulesLhs):
    rhsList=[]
    i=0
    sourceStr=str(source)
    for parital in partialMatches:
        destStr=str(dest[i])
        for matches in parital:
            destStr=destStr.replace(str(dest[i][matches[1]]),'['+str(matches[0])+']')
            #dest[i][matches[1]].set_label(matches[0])
            sourceStr=sourceStr.replace(str(source[matches[0]]),'('+source[matches[0]].label()+'.*?'+')')

        rhsList.append(destStr)
        i+=1
    rulesLhs.append(sourceStr)
    rulesRhs.append(rhsList)
    #rules.append((source,rhsList))

def train(orgSentences,simpleSentences,parser):
    rulesLhs=[]
    rulesRhs=[]
    noOfSentences=np.minimum(len(orgSentences),len(simpleSentences))
    for i in range(noOfSentences):
        sourceTree=parser.parse(orgSentences[i])[0]
        splitSimpleSentence=str(simpleSentences[i]).split('.')
        srcCopy=sourceTree.copy(deep=True)
        for j in range(len(sourceTree.leaves())):
            srcCopy[sourceTree.leaf_treeposition(j)] = '.*?'
        if len(splitSimpleSentence)==1:
            destTree=parser.parse(simpleSentences[i])[0]
            completeMatch, partialMatch =alignTrees(sourceTree,destTree, matchWords(sourceTree.leaves(),destTree.leaves()))
            extractRules(srcCopy,destTree,completeMatch,partialMatch,rulesRhs,rulesLhs)
        else:
            destTrees=[]
            partialMatches=[]
            sourceLeaves=(sourceTree.leaves()).copy()
            for simpleSentence in splitSimpleSentence:
                destTree = parser.parse(simpleSentence)[0]
                completeMatch, partialMatch = alignTrees(sourceTree, destTree,matchWords(sourceLeaves, destTree.leaves()))
                extractRules(srcCopy, destTree, completeMatch, partialMatch,rulesRhs,rulesLhs)
                destTrees.append(destTree)
                partialMatches.append(partialMatch)

            extractSplit(srcCopy,destTrees,partialMatches,rulesRhs,rulesLhs)
    return rulesRhs,rulesLhs


stanfordParser=stanford.nlpParser('http://localhost:9005')
orginal=pre.readFile("org.txt")
simple=pre.readFile("simple.txt")
rulesRhs,rulesLhs=train(orginal.splitlines(),simple.splitlines(),stanfordParser)
openedFile=io.open("rulesLhs.txt",'w')
for element in rulesLhs:
    openedFile.writelines(str(element).replace('\n','=')+'\n')
openedFile.close()
openedFile=io.open("rulesRhs.txt",'w')
for element in rulesRhs:
    openedFile.writelines(str(element)+'\n')
openedFile.close()
