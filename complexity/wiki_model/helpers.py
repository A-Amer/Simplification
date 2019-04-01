## English Command
#java -mx4g -cp "*" edu.stanford.nlp.pipeline.StanfordCoreNLPServer -annotators "tokenize,ssplit,pos,lemma,parse,sentiment" -port 9000 -timeout 30000

## Arabic Command
#java -Xmx4g -cp "*" edu.stanford.nlp.pipeline.StanfordCoreNLPServer \
# -serverProperties StanfordCoreNLP-arabic.properties \
# -preload tokenize,ssplit,pos,parse \
# -status_port 9005  -port 9005 -timeout 15000

from collections import Counter

ADJ=["JJ","JJR","JJS"]
ADV=["RB" ,"RBR" ,"RBS","RP","WRB"]
VB=["MD","VB","VBD","VBG","VBN","VBP","VBZ"]
NN=["NN","NNS","NNP","NNPS"]
PRN=["DT" ,"EX", "PDT" ,"PRP","PRP$","WDT","WP","WP$"]
OTHR=["CD" ,"LS","POS","SYM","TO" , "UH", "INF" , "FW"]
PUNCT=[".", ",",":","(",")"]
CLAUSE=["SBAR","SBARQ","SINV","SQ"]


def getType(tag):
   if tag in ADJ: return "ADJ"
   elif tag in ADV: return "ADV"
   elif tag in VB: return "VB"
   elif tag in NN: return "NN"
   elif tag in PRN: return  "PRN"
   elif tag in OTHR: return "OTHR"
   elif tag in PUNCT: return "PUNCT"
   else :  return tag


def isClause(tag):
    if tag in CLAUSE:
        return True
    return False


##Other
## CD - Cardinal Number
##LS - List Item Marker
##POS - Possessive Ending
## SYM - Symbol
##UH -Interjection
##TO - infinitive to
## Pronouns
##DT -Determiner
##EX -Existential There
##PDT -Predeterminer (Both)
##PRP -Pronoun , Personal
##PRP$ -Pronoun , Possessive
##WDT -WH Determiner
##WP -WH Personal Pronoun
##WP$ -WH possessive Pronoun
## Verb Tags
# MD - VERB, modal Auxiliary
# VB - Verb, base form
# VBD - Verb, past tense
# VBG - Verb, gerund or present participle
# VBN - Verb, past participle
# VBP - Verb, non-3rd person singular present
# VBZ - Verb, 3rd person singular present
##Nouns
##NN   Noun , singular or mass
##NNS  Noun , plural
##NNP  Noun , proper singular
##NNPS Noun , Proper Plural
##Conjunctions
# CC -Coordinating Conjunction
## Preposition
##IN - Subordinating Conjunction / Preposition
## Adjectives
# JJ - Adjective
# JJR - Adjective, comparative
# JJS - Adjective, superlative
## Adverb
# RB -Adverb
# RBR -Comparative Adverb
# RBS -Superlative adverb
# RP - Particle Adverb
# WRB - Wh- adverb








def defaultArabicPennTags():
    PennTags=Counter({'CC':0 , 'CD':0 , 'DT':0 , 'EX':0 , "FW":0 , "IN":0 , "JJ":0 , "JJR":0 , "JJS":0 , "LS":0,
                      "MD":0 , "NN":0 ,"NNS":0 ,"NNP":0 , "NNPS":0 , "PDT":0 , "POS":0,"PRP":0 , "PRP$":0 ,
                      "RB":0 , "RBR":0 , "RBS":0 , "RP":0 , "SYM":0 , "TO":0 , "UH":0 , "VB":0 ,"VBD":0 ,
                      "VBG":0 , "VBN":0 , "VBP":0 , "VBZ":0 , "WDT":0 , "WP":0 , "WP$":0 , "WRB":0  , "JJR":0 ,"ADJ_NUM":0,
                      "VN":0 , "NN":0 ,"DTJJ":0 ,"DTNNPS":0 , "DTNNP":0 , "DTNNS":0 , "DTNN":0 , "NOUN_QUANT":0 , "DTNOUN_QUANT":0 , "DTJJR":0 , "PUNC":0
                      })
    return PennTags
