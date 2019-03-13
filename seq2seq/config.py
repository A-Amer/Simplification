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
maxEpoch=100
gradientMax=5
learningRate=0.001
beta1=0.9
beta2=0.999
initWeights=0.1

fullOrgTxtPath='/home/amira/GP-imp/Abstractive Sum/data-simplification/wikilarge/wiki.full.aner.ori.train.src'
fullSimpleTxtPath='/home/amira/GP-imp/Abstractive Sum/data-simplification/wikilarge/wiki.full.aner.ori.train.dst'
orgTrainingFilePath= './data/train/Org/*.txt'
simpleTrainingFilePath= './data/train/Simple/*.txt'
orgVocabFilePath='data/orgVocab.txt'
simpleVocabFilePath='data/simpleVocab.txt'
checkpointPath='check.tar'