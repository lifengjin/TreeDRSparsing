import torch
import torch.nn as nn

class encoder(nn.Module):
    def __init__(self, args):
        super(encoder, self).__init__()
        self.args = args
        self.lstm = nn.LSTM(args.input_dim, args.bilstm_hidden_dim, num_layers=args.bilstm_n_layer, bidirectional=True)

    def forward(self, input_t, train=True):
        hidden_t = self.inithidden()
        if not test:
            self.lstm.dropout = self.args.dropout_f
        else:
            self.lstm.dropout = 0
        output_t, _ = self.lstm(input_t.unsqueeze(1), hidden_t)
        return output_t

    def inithidden(self):
        if self.args.gpu:
            result = (torch.zeros(2*self.args.bilstm_n_layer, 1, self.args.bilstm_hidden_dim, requires_grad=True).cuda(),
                torch.zeros(2*self.args.bilstm_n_layer, 1, self.args.bilstm_hidden_dim, requires_grad=True).cuda())
            return result
        else:
            result = (torch.zeros(2*self.args.bilstm_n_layer, 1, self.args.bilstm_hidden_dim, requires_grad=True),
                torch.zeros(2*self.args.bilstm_n_layer, 1, self.args.bilstm_hidden_dim, requires_grad=True))
            return result

class comb_encoder(nn.Module):
    def __init__(self, args):
        super(comb_encoder, self).__init__()
        self.args = args
        self.lstm = nn.LSTM(args.input_dim, args.bilstm_hidden_dim, num_layers=args.bilstm_n_layer, bidirectional=True)

    def forward(self, input_t, comb, train=True):
        hidden_t = self.inithidden()
        if train:
            self.lstm.dropout = self.args.dropout_f
        else:
            self.lstm.dropout = 0
        output_t, hidden_t = self.lstm(input_t.unsqueeze(1), hidden_t)

        encoder_rep = []
        for i in range(len(comb)):
            if comb[i][0] < i: # skpped becuase there are already a same lemma ahead
                continue
            encoder_rep.append([])
            for idx in comb[i]:
                encoder_rep[-1].append(output_t[idx])
            encoder_rep[-1] = (torch.sum(torch.cat(encoder_rep[-1]),0)/(len(comb[i])+1)).unsqueeze(0)
        encoder_rep = torch.cat(encoder_rep[1:-1], 0) ## <s> </s>

        return output_t, encoder_rep, hidden_t

    def inithidden(self):
        if self.args.gpu:
            result = (torch.zeros(2*self.args.bilstm_n_layer, 1, self.args.bilstm_hidden_dim, requires_grad=True).cuda(),
                torch.zeros(2*self.args.bilstm_n_layer, 1, self.args.bilstm_hidden_dim, requires_grad=True).cuda())
            return result
        else:
            result = (torch.zeros(2*self.args.bilstm_n_layer, 1, self.args.bilstm_hidden_dim, requires_grad=True),
                torch.zeros(2*self.args.bilstm_n_layer, 1, self.args.bilstm_hidden_dim, requires_grad=True))
            return result