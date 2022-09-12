import torch
import torch.nn as nn
import numpy as np

class LSTM(nn.Module):
    def __init__(self, config):
        super(LSTM, self).__init__()
        self.input_size = config['network']['input_size']
        self.output_size = config['network']['output_size']
        self.hidden_size = config['network']['hidden_size']
        self.n_layers = config['network']['n_layers']
        self.learning_rate = config['train']['learning_rate']
        self.dropout = config['train']['dropout']

        # RNN Architecture
        self.embedding = nn.Embedding(self.input_size, self.hidden_size)
        self.encoder = nn.LSTM(self.hidden_size, self.hidden_size,
                               self.n_layers, dropout=self.dropout)
        self.decoder = nn.Linear(self.hidden_size, self.output_size)

    def forward(self, inp, hidden):
        batch_size = inp.size(1)
        seq_len = inp.size(0)
        output = self.embedding(inp)
        output, hidden = self.encoder(output.view(seq_len, batch_size, -1), hidden)
        output = self.decoder(output)
        m = torch.nn.Softmax(dim=2)
        #distribution = m(output.detach().cpu().flatten()).numpy()
        distribution = m(output)
        return output, hidden, distribution

    def sample(self, inp, hidden, seq_length, temperature, device):
        m = torch.nn.Softmax(dim=0)
        batch_size = inp.size(1)
        inp = inp.to(device)
        hidden_sample = (hidden[0].clone().to(device), hidden[1].clone().to(device))

        with torch.no_grad():
            self.eval()
            sample = torch.zeros((seq_length+1, batch_size), dtype=torch.long)
            sample[0, :] = inp[-1, :]
            
            if inp.size(0) > 1:
                _0, hidden_sample, _1 = self.forward(inp[:-1], hidden_sample)
            else:
                _0, hidden_sample, _1 = self.forward(inp, hidden_sample)

            for kb in range(batch_size):

                hiddent = (hidden_sample[0][:, kb:kb+1, :].contiguous(),
                           hidden_sample[1][:, kb:kb+1, :].contiguous())

                x = torch.LongTensor([[inp[-1, kb].item()]]).to(device)

                for k in range(seq_length):
                    x, hiddent, _ = self.forward(x, hiddent)
                    if temperature < 1.e-8:
                        #print(x.shape)
                        top_i = torch.argmax(x.flatten()).item()
                    else:
                        output_dist = x.data.view(-1).div(temperature)  # .exp()
                        output_dist = m(output_dist)
                        y = torch.multinomial(output_dist, 1)
                        top_i = y[0].item()
                    sample[k+1, kb] = top_i
                    x = torch.LongTensor([[top_i]]).to(device)
        sample = sample.to(device)
        self.train()
        return sample


    def init_hidden(self, batch_size, device):
        hidden = (torch.zeros(self.n_layers, batch_size, self.hidden_size).to(device),
                  torch.zeros(self.n_layers, batch_size, self.hidden_size).to(device))
        return hidden

    def detach_hidden(self, hidden, device, clone=False):
        if clone:
            return (hidden[0].detach().clone().to(device), hidden[1].detach().clone().to(device))
        else:
            return (hidden[0].detach().to(device), hidden[1].detach().to(device))


if __name__ == '__main__':

    alphabet = [' ', '!', '$', "'", ',', '-', '.', '0', '1', '2', '3', '4', '5', '6', '7', '8', '9',
                ':', ';', '?', 'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n',
                'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z', '£', '«', '»']

    print(len(alphabet))

    config = dict()

    config['network'] = dict()

    config['network']['model'] = 'LSTM'
    config['network']['input_size'] = len(alphabet)
    config['network']['output_size'] = len(alphabet)
    config['network']['hidden_size'] = 100
    config['network']['n_layers'] = 3

    config['train'] = dict()
    # normal training
    config['train']['learning_rate'] = 0.01
    config['train']['batch_size'] = 10
    config['train']['seq_length'] = 100  # args.rnn_seq_len
    # dreaming training
    config['train']['learning_rate_dream'] = 0.01
    config['train']['batch_size_dream'] = 1
    config['train']['seq_length_dream'] = 100  # args.rnn_seq_len_dream
    config['train']['temperature_dream'] = 1.
    # general
    config['train']['dropout'] = 0.
    config['train']['cuda'] = True

    cuda = config['train']['cuda']
    if torch.cuda.is_available() and cuda:
        device = torch.device('cuda:0')
        print('Using CUDA')
    else:
        device = torch.device('cpu')
        print('Using CPU')

    batch_size = config['train']['batch_size']
    seq_length = config['train']['seq_length']
    batch_size_dream = config['train']['batch_size_dream']
    seq_length_dream = config['train']['seq_length_dream']
    lr = config['train']['learning_rate']
    lr_dream = config['train']['learning_rate_dream']
    temperature_dream = config['train']['temperature_dream']

    model = LSTM(config).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = torch.nn.CrossEntropyLoss()

    # ------------------------

    batch_size = 10
    seq_length = 20

    sample_length = 12
    temperature = 0.1

    x = torch.randint(len(alphabet), (seq_length, batch_size)).to(device)
    print(x.size())

    hidden = model.init_hidden(batch_size, device)

    print(model.sample(x, hidden, sample_length, temperature, device))