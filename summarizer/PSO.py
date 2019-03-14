import numpy as np
import random
#class
class Particle(object):
    def __init__(self,intialPos):
        self._position = intialPos  # particle position
        self._velocity = []  # particle velocity
        self._posBest = intialPos # best position individual
        self._costBest = -1  # best error individual
        self._cost = -1  # cost individual
        self._r1=random.uniform(0,1)
        self._r2 = random.uniform(0, 1)
        self._c1=1.8#personal accelration
        self._c2=2#social accelration
        self._inertiaCoeff=0.9
        for i in range(len(intialPos)):
            self._velocity.append(random.uniform(-1, 1))
        self._velocity=np.array(self._velocity)

    def costFunc(self,similarityMatrix,infScores):
        summarySentenceIndex=self._position[self._position==1]
        cost=0
        for i in summarySentenceIndex:
            for j in summarySentenceIndex[int(i):]:
                cost+=similarityMatrix[int(i)][int(j)]*infScores[int(i)]
        return cost


    def evaluate(self,simMatrix,infScores):
        self._cost=self.costFunc(simMatrix,infScores)
        if self._cost>self._costBest:
            self._costBest=self._cost
            self._posBest=self._position
        return self._cost

    def updatePosition(self):
        self._position=self._position+self._velocity
        self._position[self._position< 1]=0
        self._position[self._position >= 1]=1


    def updateVelocity(self,globalBest):
        self._velocity=self._velocity*self._inertiaCoeff+(self._c1*self._r1)*(self._posBest-self._position)+(self._c2*self._r2)*(globalBest-self._position)




class PSO(object):
    def __init__(self,  sentences,similarityMatrix,infScores, maxLen, numberLocations, trialLimit, mfe):
        self._noLocations = numberLocations
        self._trialLimit = trialLimit
        self._simMat=similarityMatrix
        self._infScores=infScores
        self._mfe = mfe
        self._sentences =list(zip(range(len(sentences)),sentences))
        self._maxLen = maxLen
        self._population=[]
        self._globalBest=-1
        self._globalBestSol=np.zeros(len(sentences))

    def createRandomIndivdual(self):
        randomScores = np.random.rand(len(self._sentences))
        scoredSentences = list(zip(self._sentences, randomScores))
        sortedSentences = sorted(scoredSentences, key=lambda tup: tup[1], reverse=True)
        avaliableWords=self._maxLen
        candSolution=np.zeros(len(self._sentences))
        for sentence in sortedSentences:
            candSolution[sentence[0][0]]=1
            avaliableWords-=len(sentence[0][1])
            if avaliableWords<0:
                break
        return Particle(candSolution)

    def intializePopulation(self):
        for i in range(self._noLocations):
            self._population.append(self.createRandomIndivdual())
        return
    def getBestSolution(self):
        self.intializePopulation()
        for i in range(self._trialLimit):
            for particle in self._population:
                cost=particle.evaluate(self._simMat,self._infScores)
                if cost >self._globalBest:
                    self._globalBest=cost
                    self._globalBestSol=particle._position
            #for particle in self._population:
                particle.updateVelocity(self._globalBest)
                particle.updatePosition()

        return self._globalBestSol
