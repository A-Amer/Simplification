import  nlpParser as stanford
from nltk.tokenize.treebank import TreebankWordDetokenizer
import ast
import re
import nltk.tree as tree
import io

ruleLhs=[]
ruleRhs=[]
def intializeParser():
    with open("rulesLhs.txt") as f:
        for line in f:
            ruleLhs.append(re.compile(line.replace('\n','').replace('(','\(').replace(')','\)')))

    with open("rulesRhs.txt") as f:
        for line in f:
            if line.__contains__("'"):
                ruleRhs.append(ast.literal_eval(line.replace('\n','')))
            else:
                ruleRhs.append(line.replace('\n',''))
    stanfordParser = stanford.nlpParser('http://localhost:9005')

    return stanfordParser

def matchSentence(sentence,sTree):
    indices = [i for i, s in enumerate(ruleLhs) if s.fullmatch(sentence)!=None]
    rule=None
    if indices.__len__()!=0:
        rules=ruleRhs[indices[0]]
    else:
        return str(sTree)
    if type(rules) is str:
        tobeReplaced = re.findall(r'\[\(.*?\)\]', rules)
        for replaceStr in tobeReplaced:
            replaceTree = sTree[ast.literal_eval(replaceStr.replace('[', '').replace(']', ''))]
            replacement = str(replaceTree)
            # replacement=matchSentence(replacement,replaceTree)
            rules = rules.replace(replaceStr, replacement)
    else:
        for i in range(len(rules)):
            tobeReplaced=re.findall(r'\[\(.*?\)\]',rules[i])
            for replaceStr in tobeReplaced:
                replaceTree=sTree[ast.literal_eval(replaceStr.replace('[','').replace(']',''))]
                replacement=str(replaceTree)
                replacement=matchSentence(replacement,replaceTree)
                rules[i]=rules[i].replace(re.escape(replaceStr),replacement)
    return rules

parser=intializeParser()
sentence="عمر الذي يلعب الكرة رجل نشيط"
sourceTree=parser.parse(sentence)[0]
output=matchSentence(str(sourceTree).replace('\n',''),sourceTree)
outputTree=tree.Tree.fromstring(output)
x=TreebankWordDetokenizer()
s=x.detokenize(outputTree.leaves()).replace(' ب ',' ب').replace(' ل ال',' لل').replace(' س ',' س')
print(s)





