import  nlpParser as stanford
from nltk.tokenize.treebank import TreebankWordDetokenizer
import ast
import re
import nltk.tree as tree

ruleLhs=[]
ruleRhs=[]
def intializeParser():
    with open("rulesLhs.txt") as f:
        for line in f:
            ruleLhs.append(re.compile(line.replace('\n','').replace('(','\(').replace(')','\)').replace('=','\n')))

    with open("rulesRhs.txt") as f:
        for line in f:
            if line.__contains__("'"):
                ruleRhs.append(ast.literal_eval(line.replace('\n','').replace('=','\n')))
            else:
                ruleRhs.append(line.replace('\n','').replace('=','\n'))
    stanfordParser = stanford.nlpParser('http://localhost:9005')

    return stanfordParser

def matchSentence(sentence,sTree):
    indices = [i for i, s in enumerate(ruleLhs) if s.fullmatch(sentence)!=None]
    rules=None
    if indices.__len__()!=0:
        rules=ruleRhs[indices[0]]
    else:
        return str(sTree)
    if type(rules) is str:
        tobeReplaced = re.findall(r'\[\(.*?\)\]', rules)
        for replaceStr in tobeReplaced:
            replaceTree = sTree[ast.literal_eval(replaceStr.replace('[', '').replace(']', ''))]
            replacement = str(replaceTree)
            replacement=matchSentence(replacement,replaceTree)
            rules = rules.replace(replaceStr, replacement)
    else:
        for i in range(len(rules)):
            tobeReplaced=re.findall(r'\[\(.*?\)\]',rules[i])
            for replaceStr in tobeReplaced:
                replaceTree=sTree[ast.literal_eval(replaceStr.replace('[','').replace(']',''))]
                replacement=str(replaceTree)
                replacement=matchSentence(replacement,replaceTree)
                rules[i]=rules[i].replace(replaceStr,replacement)
    return rules

parser=intializeParser()
sentence="الكرة يلعب بها عمر"
sourceTree=parser.parse(sentence)[0]
outputArray=matchSentence(str(sourceTree),sourceTree)
if type(outputArray) is str:
    outputTree = tree.Tree.fromstring(outputArray)
    x = TreebankWordDetokenizer()
    s = x.detokenize(outputTree.leaves()).replace(' ب ', ' ب').replace(' ل ال', ' لل').replace(' س ', ' س')
    print(s)
else:
    for output in outputArray:
        outputTree = tree.Tree.fromstring(output)
        x=TreebankWordDetokenizer()
        s=x.detokenize(outputTree.leaves()).replace(' ب ',' ب').replace(' ل ال',' لل').replace(' س ',' س')
        print(s)





