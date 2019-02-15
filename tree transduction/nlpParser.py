from nltk.parse.corenlp import CoreNLPParser
from nltk.parse.corenlp import  CoreNLPServer


class nlpParser(object):
    def __init__(self, port):
        self._parser= CoreNLPParser(port)
        #self._posTagger = CoreNLPParser(port, tagtype='pos')
        #self._nerTagger=CoreNLPParser(port,tagtype='ner')

    def parse(self,text):
        return  list(self._parser.raw_parse(text))
    def parseSentences(self,text):
        return list(self._parser.raw_parse_sents(text))

    '''def posTag(self,text):
        return list(self._posTagger.tag(self._parser.tokenize(text)))

    def nerTag(self,text):
        return list(self._nerTagger.tag(self._parser.tokenize(text)))'''