import torch
import torch.nn as nn
import torch.nn.functional as f

import config
import numpy as np

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class encoder(nn.Module):
    def __init__(self):
        super(encoder, self).__init__()

        # A simple lookup table that stores embeddings of a fixed dictionary and size
        self.embedding = nn.Embedding(config.vocabSizeEnc, config.embDimEnc)

        #recurrent layer using LSTM
        self.rnn = nn.LSTM(config.embDimEnc,
                           config.hiddenDimEnc,
                           num_layers=config.nLayersEnc,
                           bidirectional=config.bidirectionalEnc,
                           dropout=config.dropoutEnc)

        self.dropout = nn.Dropout(config.dropoutEnc)

    def forward(self, input):
        #applying embedding to input sequence
        embedded = self.dropout(self.embedding(input))

        #Inputs: input, (h_0, c_0)
        #(h_0, c_0): tensor containing the initial hidden/cell state for each element in the batch
        #If (h_0, c_0) is not provided, both h_0 and c_0 default to zero
        #embedded: input of shape (seq_len, batch, input_size)
        # outputs: tensor containing the output features (h_t) from the last layer of the LSTM
        # hidden/cell: tensor containing the hidden/cell state for t = seq_len
        outputs, (hidden, cell) = self.rnn(embedded)
    
        return outputs, hidden, cell


class decoderSimp(nn.Module):
    def __init__(self):
        super(decoderSimp, self).__init__()

        self.embedding = nn.Embedding(config.vocabSizeDec, config.embDimDec)
        self.rnn = nn.LSTM(config.embDimDec,
                           config.hiddenDimDec,
                           num_layers=config.nLayersDec,
                           bidirectional=config.bidirectionalDec,
                           dropout=config.dropoutDec)
        self.dropout = nn.Dropout(config.dropoutDec)

        self.neuNet = nn.Linear(config.hiddenDimDec * 2, config.hiddenDimDec, bias=False)
        self.out = nn.Linear(config.hiddenDimDec, config.vocabSizeDec, bias=False)

    def forward(self, input, encoderStates, hidden, cell):

        #input = input.unsqueeze(0)
        embedded = self.dropout(self.embedding(input))
        embedded=embedded.unsqueeze(0)
        output, (hidden, cell) = self.rnn(embedded, (hidden, cell))

        hiddenUpperLayer = torch.t(hidden.squeeze(1)[1].unsqueeze(0))
        atten = torch.mm(encoderStates, hiddenUpperLayer)
        atten = f.softmax(atten)
        context = torch.mm(torch.t(encoderStates), atten)
        nnInput = torch.cat((hiddenUpperLayer, context))

        prediction = f.softmax(self.out(f.tanh(self.neuNet(torch.t(nnInput))))) #neural network layer

        return prediction, hidden, cell


class seq2seq(nn.Module):
    def __init__(self,eosToken,sosToken):
        super().__init__()

        self.encoder = encoder()
        self.decoder = decoderSimp()
        self.sosToken=torch.tensor(sosToken, dtype=torch.long)
        self.eosToken=torch.tensor(eosToken, dtype=torch.long)
        assert config.hiddenDimEnc == config.hiddenDimDec, "Hidden dimensions of encoder and decoder must be equal!"
        assert config.nLayersEnc == config.nLayersDec, "Encoder and decoder must have equal number of layers!"

    def forward(self, src, trgLen):

            outputs = torch.zeros(trgLen)

            encoderStates, hidden, cell = self.encoder(src)
            encoderStates = encoderStates.squeeze(1)

            input = self.sosToken.unsqueeze(0)
            outputs[0] = input

            t=1

            while input != self.eosToken and t < trgLen:
                output, hidden, cell = self.decoder(input, encoderStates, hidden, cell)
                outputs[t] = output.max(1)[1]
                input = output.max(1)[1]
                t = t+1

            return outputs

