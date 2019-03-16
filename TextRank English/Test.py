import numpy as np
import nltk
from collections import Counter
from nltk.util import ngrams
import rouge

systemFile = open("system.txt")
referenceFile = open("reference.txt")

referenceSummary = referenceFile.read().lower() #to lower
systemSummary = systemFile.read().lower() ##to lower


def rougeN(referenceSummary,  systemSummary, n=2):
    refNGrams = Counter(ngrams(referenceSummary.split(), n))
    #print(refNGrams)

    sysNGrams=Counter(ngrams(systemSummary.split(), n))
    #print(sysNGrams)

    overlappingNGrams=0
    sumRefNGrams=0
    recall=0

    for (k,v) in refNGrams.items():
        sumRefNGrams+=v
        if k in sysNGrams:
            overlappingNGrams+=min(v, sysNGrams[k])


    recall=overlappingNGrams/sumRefNGrams
    print(recall)
    return recall


import collections


def lcs(s1, s2):
    tokens1, tokens2 = s1.split(), s2.split()
    cache = collections.defaultdict(dict)
    for i in range(-1, len(tokens1)):
        for j in range(-1, len(tokens2)):
            if i == -1 or j == -1:
                cache[i][j] = 0
            else:
                if tokens1[i] == tokens2[j]:
                    cache[i][j] = cache[i - 1][j - 1] + 1
                else:
                    cache[i][j] = max(cache[i - 1][j], cache[i][j - 1])
    return cache[len(tokens1) - 1][len(tokens2) - 1]


def rougeL(referenceSummary, systemSummary):
    m = len(referenceSummary.split())
    print(m)
    longestCommonSub = lcs(referenceSummary, systemSummary) / m
    print(longestCommonSub)
    return longestCommonSub


# A translation using the same words (1-grams)
# as in the references tends to satisfy adequacy. The
# longer n-gram matches account for fluency

##Modified Bleu
def bleuN(referenceSummary , systemSummary , n=2):

    refNGrams = Counter(ngrams(referenceSummary.split(), n))
    # print(refNGrams)
    sysNGrams = Counter(ngrams(systemSummary.split(), n))
    # print(sysNGrams)

    overlappingNGrams = 0
    sumSysNGrams = 0
    precision = 0

    for (k, v) in sysNGrams.items():
        sumSysNGrams += v
        if k in refNGrams:
            overlappingNGrams += min(v, refNGrams[k])

    precision = overlappingNGrams / sumSysNGrams
    print(precision)
    return precision













print("Rouge 1-2-3")
rougeN(referenceSummary,systemSummary, 1)
rougeN(referenceSummary,systemSummary, 2)
rougeN(referenceSummary,systemSummary, 3)
#print("LCS")
##rougeL(referenceSummary , systemSummary)
print("Bleu")
bleuN(referenceSummary,systemSummary)
##Comparing my results

evaluator = rouge.Rouge( metrics=['rouge-n', 'rouge-l'],
                           max_n=3,
                           limit_length=False,
                           length_limit=100,
                           length_limit_type='words',
                           alpha=0.5, # Default F1_score
                           weight_factor=1.2,
                           stemming=False)
scores=evaluator.get_scores([systemSummary], [referenceSummary])
print(scores)



from nltk.translate.bleu_score import sentence_bleu
score = sentence_bleu( [referenceSummary.split()] ,  systemSummary.split() , weights=(0, 1, 0, 0))
print("Bleu Score" , score)



