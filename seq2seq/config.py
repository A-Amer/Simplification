#encoder parameters
vocabSizeEnc = 1000
embDimEnc = 300
hiddenDimEnc = 256
nLayersEnc = 2
bidirectionalEnc = False
dropoutEnc = 0.2

#decoder parameters
vocabSizeDec = 1000
embDimDec = 300
hiddenDimDec = 256
nLayersDec = 2
bidirectionalDec = False
dropoutDec = 0.2

vocabSize=50000
maxEpoch=100
gradientMax=5
fullOrgTxtPath='/home/amira/GP-imp/Abstractive Sum/data-simplification/wikilarge/wiki.full.aner.ori.train.src'
fullSimpleTxtPath='/home/amira/GP-imp/Abstractive Sum/data-simplification/wikilarge/wiki.full.aner.ori.train.dst'
orgTrainingFilePath= '/home/amira/GP-imp/Abstractive Sum/data/train/Org/*.txt'
simpleTrainingFilePath= '/home/amira/GP-imp/Abstractive Sum/data/train/Simple/*.txt'
orgVocabFilePath='/home/amira/GP-imp/Abstractive Sum/data/orgVocab.txt'
simpleVocabFilePath='/home/amira/GP-imp/Abstractive Sum/data/simpleVocab.txt'
checkpointPath=''