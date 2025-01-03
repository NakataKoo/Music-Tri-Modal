import math
import numpy as np
import random

import torch
import torch.nn as nn
import torch.nn.functional as F

class SequenceClassification(nn.Module):
    def __init__(self, midibert, class_num, hs, da=128, r=4):
        super(SequenceClassification, self).__init__()
        self.midibert = midibert
        midi_size = 10
        self.midi_compressor = nn.Linear(midi_size, 1, bias=False)
        self.attention = SelfAttention(hs, da, r)
        self.classifier = nn.Sequential(
            nn.Linear(hs*r, 256),
            nn.ReLU(),
            nn.Linear(256, class_num)
        )

    def forward(self, x, attn, layer):             # x: (batch, 512token, 4)
        #x = self.midibert(x, attn, output_hidden_states=True)   # (batch, 512token, 512dim)
        #x = self.midibert.encode_midi(x)

        # (batch, midi_size, 512, 4) -> (batch, 512, 4, midi_size)
        x = x.permute(0, 2, 3, 1)
        x = x.float()

        # (batch, 512, 4, midi_size) -> (batch, 512, 4, 1)
        x = self.midi_compressor(x)

        # (batch, 512, 4, 1) -> (batch, 512, 4)
        x = x.squeeze(-1)

        x = self.midibert.midibert(x, attn, output_hidden_states=True)  # (batch, 512, hs)

        #y = y.last_hidden_state         # (batch_size, seq_len, 768)
        x = x.hidden_states[layer]
        attn_mat = self.attention(x)        # attn_mat: (batch, r, 512)
        m = torch.bmm(attn_mat, x)          # m: (batch, r, 768)
        flatten = m.view(m.size()[0], -1)   # flatten: (batch, r*768)
        res = self.classifier(flatten)      # res: (batch, class_num)
        return res


class SelfAttention(nn.Module):
    def __init__(self, input_dim, da, r):
        '''
        Args:
            input_dim (int): batch, seq, input_dim
            da (int): number of features in hidden layer from self-attn
            r (int): number of aspects of self-attn
        '''
        super(SelfAttention, self).__init__()
        self.ws1 = nn.Linear(input_dim, da, bias=False)
        self.ws2 = nn.Linear(da, r, bias=False)

    def forward(self, h):
        attn_mat = F.softmax(self.ws2(torch.tanh(self.ws1(h))), dim=1)
        attn_mat = attn_mat.permute(0,2,1)
        return attn_mat
    

class Classification(nn.Module):
    def __init__(self, midibert, class_num, hs, da=128, r=4):
        super().__init__()
        self.midibert = midibert
        self.attention = nn.MultiheadAttention(embed_dim=hs, num_heads=4)
        self.classifier = nn.Sequential(
            nn.Linear(hs, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, class_num)
        )


    def forward(self, x, attn, layer):             # x: (batch, midi_dim, 512token, 4)
        x = self.midibert.encode_midi(x)          # (batch, hs)
        attentioin_output, _ = self.attention(x, x, x)
        logits = self.classifier(attentioin_output)
        return logits