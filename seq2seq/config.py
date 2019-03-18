#encoder parameters
vocabSizeEnc = 0
embDimEnc = 300
hiddenDimEnc = 256
nLayersEnc = 2
bidirectionalEnc = False
dropoutEnc = 0.2

#decoder parameters
vocabSizeDec = 0
embDimDec = 300
hiddenDimDec = 256
nLayersDec = 2
bidirectionalDec = False
dropoutDec = 0.2
beamSize=3

#training parameters
vocabSize=50000
maxEpoch=1
gradientMax=5
learningRate=0.001
beta1=0.9
beta2=0.999
initWeights=0.1
batchSize=64
batchPerFile=16

#relevance model parameters
nLayersRelEnc=1
nLayersRelDec=1

fullOrgTxtPath='./data/wiki.full.aner.train.src'
fullSimpleTxtPath='./data/wiki.full.aner.train.dst'
orgTrainingFilePath= './data/train/Org'
simpleTrainingFilePath= './data/train/Simple'
orgVocabFilePath='data/orgVocab.txt'
simpleVocabFilePath='data/simpleVocab.txt'
checkpointPath='check.tar'