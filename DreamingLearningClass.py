import sys
from Models import LSTM
import numpy as np
import torch
import copy
import tqdm
import os


class DreamingLearningClass:
    def __init__(self,
                 model_config,
                 initial_state_dict=None):

        self.config = model_config

        self.model_type = self.config['network']['model_type']

        #if self.config['train']['temperature_dream'] is None:
        #    self.dream = False
        #    self.temperature_dream = -1.
        #elif self.config['train']['temperature_dream'] < 0.0:
        #    self.dream = False
        #    self.temperature_dream = -1.
        #else:
        #    self.dream = True
        #    self.temperature_dream = self.config['train']['temperature_dream']

        cuda = self.config['train']['cuda']
        if torch.cuda.is_available() and cuda:
            self.device = torch.device('cuda:0')
            print('Using CUDA')
        else:
            self.device = torch.device('cpu')
            print('Using CPU')


        if self.model_type == 'LSTM':
            self.model = LSTM(self.config).to(self.device)
        else:
            raise ValueError(f'{self.model_type} not yet implemented')

        if initial_state_dict is not None:
            self.model.load_state_dict(copy.deepcopy(initial_state_dict))

        self.lr = self.config['train']['learning_rate']

        if self.config['train']['optimizer'] == 'Adam':
            self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)
        elif self.config['train']['optimizer'] == 'SGD':
            self.optimizer = torch.optim.SGD(self.model.parameters(), lr=self.lr)
        else:
            raise ValueError('Wrong Optimizer')

        if self.config['train']['criterion'] == 'CrossEntropyLoss':
            self.criterion = torch.nn.CrossEntropyLoss()
        else:
            raise ValueError('Wrong Criterion')

        #self.batch_size = self.config['train']['batch_size']
        #self.seq_length = self.config['train']['seq_length']

        #self.lr = self.config['train']['learning_rate']

        # If dreaming parameters are not included in the config dictionary, the following variables will be None
        self.seq_length_dream = self.config['train'].get('seq_length_dream')
        self.batch_size_dream = self.config['train'].get('batch_size_dream')
        self.lr_dream = self.config['train'].get('learning_rate_dream')
        self.temperature_dream = self.config['train'].get('temperature_dream')

        #self.hidden = self.model.init_hidden(self.batch_size, self.device)
        self.hidden = None

        self.train_loss_seq = []
        self.valid_loss_seq = []

        self.distr = None

    #### SETTERS AND GETTERS ########

    def set_temperature(self, temperature):
        self.temperature_dream = temperature

    def get_temperature(self):
        return self.temperature_dream

    def set_seq_len_dream(self, seq_length_dream):
        self.seq_length_dream = seq_length_dream

    def get_seq_len_dream(self):
        return self.set_seq_len_dream

    def set_lr_dream(self, lr_dream):
        self.lr_dream = lr_dream

    def get_lr_dream(self):
        return self.lr_dream

    ################################

    def generate(self, inp, hidden=None):
        batch_size = inp.size(1)

        if hidden is None:
            hidden = self.model.init_hidden(batch_size, self.device)

        sample = self.model.sample(inp.to(self.device),
                                   hidden,
                                   self.seq_length_dream,
                                   self.temperature_dream,
                                   self.device)

        return sample


    def validate(self, inp, hidden=None, get_distr=False, get_out=False):

        self.model.eval()
        losst = self._normal_train(inp,
                                   hidden=hidden,
                                   validation=True,
                                   get_distr=get_distr)
        if get_out is False:
            self.valid_loss_seq.append(losst.item())
        else:
            return losst


    def train(self,
              inp,
              hidden=None,
              dream_init_seq=None,
              temperature_dream=None,
              seq_length_dream=None,
              lr_dream=None,
              keep_dream_hidden=False,
              dream_first=False,
              get_distr=False):

        # dream_init_seq: torch tensor (time x batch) containing the initial sequence to produce the dreaming
        #                 sequence (if None, the initial sequence will be [[0]])
        # temperature_dream: temperature for dreaming sampling. If None, the object parameter self.temperature_dream
        #                    will be used
        # seq_length_dream: sequence length for dreaming sampling. If None, the object parameter
        #                   self.seq_length_dream will be used
        # lr_dream: learning rate for dreaming training. If None, the object parameter self.lr_dream will be used
        #
        # keep_dream_hidden: if True, the hidden state after dreaming will be kept for the next normal training
        #
        # dream_first: if True, the train will be first dreaming and then normal training.
        #
        # get_distr: if True, the probability distribution at the end of the normal train will be stored in
        #            self.distr

        self.model.train()

        dream = False
        if temperature_dream is not None:
            if temperature_dream >= 0.0:
                self.temperature_dream = temperature_dream
                dream = True
        elif self.temperature_dream is not None:
            if self.temperature_dream >= 0.0:
                dream = True

        if dream:
            if seq_length_dream is None:
                if self.seq_length_dream is None:
                    raise ValueError('Dreaming is active but no dream sequence length has been set')
            else:
                self.seq_length_dream = seq_length_dream

            if lr_dream is None:
                if self.lr_dream is None:
                    raise ValueError('Dreaming is active but no dream learning step has been set')
            else:
                self.lr_dream = lr_dream

        inp = inp.to(self.device)

        if dream:
            if dream_init_seq is None:
                inp_dream = torch.LongTensor([[0]*self.batch_size_dream]).view(1,-1)
            else:
                inp_dream = dream_init_seq
            inp_dream = inp_dream.to(self.device)

        if dream and (dream_first is True):
            _ = self._dream_train(inp_dream, clone_hidden=True, hidden=None)

        losst = self._normal_train(inp,
                                   hidden=hidden,
                                   validation=False,
                                   get_distr=get_distr)

        self.train_loss_seq.append(losst.item())

        if dream and (dream_first is False):
            _ = self._dream_train(inp_dream, clone_hidden=True, hidden=None)



    def _dream_train(self, inp, clone_hidden=True, hidden=None, keep_dream_hidden=False):

        batch_size = inp.size(1)

        if hidden is None:
            hidden = self.model.init_hidden(batch_size, self.device)
        hidden_dream = self.model.detach_hidden(hidden, self.device, clone=clone_hidden)

        sample = self.model.sample(inp.to(self.device),
                                   hidden_dream,
                                   #self.model.init_hidden(batch_size, self.device),
                                   self.seq_length_dream,
                                   self.temperature_dream,
                                   self.device)

        self.model.train()
        self.optimizer.zero_grad()
        
        # ATTENZIONE: hidden dream andrebbe perfezionato...
        if hidden is None:
            hidden = self.model.init_hidden(batch_size, self.device)
        hidden_dream = self.model.detach_hidden(hidden, self.device, clone=clone_hidden)
        
        out, hidden_dream, _1 = self.model(sample[:-1], hidden_dream)
        #m = torch.nn.Softmax(dim=2)
        target = sample[1:]
        loss = self.criterion(out.permute(0, 2, 1), target)
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = self.lr_dream
        loss.backward()
        self.optimizer.step()
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = self.lr

        if keep_dream_hidden:
            self.hidden = self.model.detach_hidden(hidden_dream, self.device, clone=True)

        return loss

    def _normal_train(self, inp, hidden=None, validation=False, get_distr=False):
        if validation:
            self.model.eval()
        else:
            self.model.train()
            self.optimizer.zero_grad()

        batch_size = inp.size(1)

        inp0 = inp[:-1]
        target = inp[1:]

        if hidden is None:
            #self.hidden = self.model.init_hidden(batch_size, self.device)
            hidden = self.model.init_hidden(batch_size, self.device)
        else:
            hidden = self.model.detach_hidden(hidden, self.device, clone=True)
        #    self.hidden = hidden

        #self.hidden = self.model.detach_hidden(self.hidden, self.device, clone=False)


        

        inp0 = inp0.to(self.device)
        target = target.to(self.device)
        out, hidden, distr = self.model(inp0, hidden)

        if get_distr:
            distr = distr.detach()
            self.distr = distr / torch.unsqueeze(distr.sum(2), 2)

        loss = self.criterion(out.permute(0, 2, 1), target)
        
        if validation:
            self.model.train()
        else:
            loss.backward()
            self.optimizer.step()
            self.hidden = self.model.detach_hidden(hidden, self.device, clone=True)

        return loss




if __name__ == '__main__':

    import sys

    sys.path.append('../Corpy/')

    # %%

    from corpy import Corpy, load_corpy
    import glob
    import torch
    import copy
    from IPython.display import display, clear_output
    import matplotlib.pyplot as plt
    import numpy as np

    from Models import LSTM
    #from Dr import Dreaming_NN


    # %%

    # %%

    def ma(x, wlen):
        x = np.array(x)
        fr = len(x) - wlen
        if fr < 0:
            fr = 0
        mean = x[fr:].mean()
        std = x[fr:].std()
        return mean, std


    # %%

    # %%

    # %% md

    ### Loading textual files

    # %%

    book_files = glob.glob('../Dreaming and Text/Books/*.txt')
    book_files = sorted(book_files)
    book_files, len(book_files)

    # %%

    books = []
    for bf in book_files:
        with open(bf, 'r') as f:
            books.append(f.read())

    # %%

    books_dict = dict()
    for k, name in enumerate(book_files):
        ind_in = name.find('/Books/')
        nname = name[ind_in+16:-4]
        section = name[ind_in+7]
        ind_dash = nname.find(' - ')
        ind_in = nname.find('EDITED')
        author = nname[:ind_dash]
        title = nname[ind_dash + 3:]
        books_dict[k] = dict()
        books_dict[k]['author'] = author
        books_dict[k]['title'] = title
        books_dict[k]['text'] = books[k]
        #if int(name[8]) == 1:
        if int(section) == 1:
            books_dict[k]['section'] = 'Training'
        elif int(section) == 2:
        #elif int(name[8]) == 2:
            books_dict[k]['section'] = 'Validation'
        elif int(section) == 3:
        #elif int(name[8]) == 3:
            books_dict[k]['section'] = 'Test'

    # %%

    for ind in books_dict.keys():
        print(ind + 1)
        print(books_dict[ind]['author'])
        print(books_dict[ind]['title'])
        print(books_dict[ind]['section'])
        print('--------------------')

    # %%

    authors = []
    titles = []
    for kb in books_dict.keys():
        authors.append(books_dict[kb]['author'])
        titles.append(books_dict[kb]['title'])

    # %%

    # %% md

    ### Corpus from Corpy class - Caharcter Level

    # %%

    corpus = Corpy(books, mode='char', text_sections=(16, 3, 1), text_sections_level='book', authors=authors,
                   titles=titles)

    # %%

    alphabet = list(corpus.item2ind.keys())

    # %%

    len(alphabet)

    # %%

    # %% md

    ### Network config

    # %%

    # %%

    config = dict()

    config['network'] = dict()

    config['network']['model_type'] = 'LSTM'
    config['network']['input_size'] = len(alphabet)
    config['network']['output_size'] = len(alphabet)
    config['network']['hidden_size'] = 100
    config['network']['n_layers'] = 3

    config = dict()

    config['network'] = dict()

    config['network']['model_type'] = 'LSTM'
    config['network']['input_size'] = len(alphabet)
    config['network']['output_size'] = len(alphabet)
    config['network']['hidden_size'] = 100
    config['network']['n_layers'] = 3

    config['train'] = dict()
    # normal training
    config['train']['learning_rate'] = 0.008
    config['train']['batch_size'] = 100
    config['train']['seq_length'] = 200  # args.rnn_seq_len
    config['train']['batch_size_validation'] = 20
    config['train']['seq_length_validation'] = 200
    # dreaming training
    config['train']['learning_rate_dream'] = 0.001
    config['train']['batch_size_dream'] = 5
    config['train']['seq_length_dream'] = 100  # args.rnn_seq_len_dream
    config['train']['temperature_dream'] = None
    # general
    config['train']['dropout'] = 0.
    config['train']['cuda'] = True
    config['train']['optimizer'] = 'Adam'
    config['train']['criterion'] = 'CrossEntropyLoss'

    config_dream = copy.deepcopy(config)
    config_dream['train']['temperature_dream'] = 0.7

    dnn_vanilla = DreamingLearningClass(config)
    init_state_dict = copy.deepcopy(dnn_vanilla.model.state_dict())
    dnn_dreaming = DreamingLearningClass(config_dream, initial_state_dict=copy.deepcopy(init_state_dict))

    namefile = '../Dreaming and Text/Text_Dreaming_Exp_01/1_Text_Dreaming_Result.pt'
    if os.path.isfile(namefile):
        result = torch.load(namefile, map_location=torch.device('cpu'))
        dnn_vanilla.model.load_state_dict(copy.deepcopy(result['Vanilla']['Last_network_dict']))
        dnn_vanilla.train_loss_seq = result['Vanilla']['Train_loss']
        dnn_vanilla.valid_loss_seq = result['Vanilla']['Validation_loss']
        best_it_vanilla = result['Vanilla']['Best_it']
        best_valid_vanilla = result['Vanilla']['Best_valid']
        best_state_vanilla = result['Vanilla']['Best_network_dict']
        dnn_dreaming.model.load_state_dict(copy.deepcopy(result['Dreaming']['Last_network_dict']))
        dnn_dreaming.train_loss_seq = result['Dreaming']['Train_loss']
        dnn_dreaming.valid_loss_seq = result['Dreaming']['Validation_loss']
        best_it_dreaming = result['Dreaming']['Best_it']
        best_valid_dreaming = result['Dreaming']['Best_valid']
        best_state_dreaming = result['Dreaming']['Best_network_dict']
        from_it = len(dnn_vanilla.train_loss_seq)

        dnn_vanilla.model.load_state_dict(copy.deepcopy(result['Vanilla']['Best_network_dict']))
        dnn_dreaming.model.load_state_dict(copy.deepcopy(result['Dreaming']['Best_network_dict']))
    else:
        print('No file', namefile)

    dnn_vanilla_normal_post = DreamingLearningClass(config, initial_state_dict=copy.deepcopy(
        result['Vanilla']['Best_network_dict']))
    # dnn_vanilla_normal_post.model.load_state_dict(copy.deepcopy(result['Vanilla']['Best_network_dict']))
    dnn_vanilla_normal_post.set_temperature(None)
    dnn_vanilla_normal_post.batch_size_dream = 1
    dnn_vanilla_normal_post.seq_length_dream = 100
    dnn_vanilla_normal_post.train_loss_seq = []
    dnn_vanilla_normal_post.valid_loss_seq = []

    corpus.reset_counter()
    last_book = corpus.get_chunk(book_sel=0, chunk_sel=0, chunk_len=1000000, chunk_mode='sequential',
                                 output_mode='code', section=2)
    len(last_book)

    #### NORMAL NET
    criterion = torch.nn.CrossEntropyLoss()
    mod_vanilla = LSTM(config)
    mod_vanilla.load_state_dict(copy.deepcopy(result['Vanilla']['Best_network_dict']))
    device = dnn_vanilla_normal_post.device
    mod_vanilla = mod_vanilla.to(device)
    #optimizer = torch.optim.SGD(mod_vanilla.parameters(), lr=config['train']['learning_rate'])
    optimizer = torch.optim.Adam(mod_vanilla.parameters(), lr=config['train']['learning_rate'])
    mod_vanilla.train()
    hidden = mod_vanilla.init_hidden(1, device)
    loss_vanilla_train_final = []


    seq_length = config['train']['seq_length']
    dnn_vanilla_normal_post.hidden = dnn_vanilla_normal_post.model.init_hidden(1, dnn_vanilla_normal_post.device)
    for k in range(0, len(last_book), seq_length):
        chunk = last_book[k:k + seq_length + 1]
        inp = torch.LongTensor(chunk).view(-1, 1)
        dnn_vanilla_normal_post.train(inp, hidden=dnn_vanilla_normal_post.hidden)

        #### NORMAL NET
        optimizer.zero_grad()
        if len(chunk) == seq_length + 1:
            inp = torch.LongTensor([chunk[:-1]]).view(-1, 1).to(device)
            target = torch.LongTensor([chunk[1:]]).view(-1, 1).to(device)
            out, hidden, _1 = mod_vanilla(inp, hidden)
            hidden = mod_vanilla.detach_hidden(hidden, device, clone=False)
            losst = criterion(out.permute(0, 2, 1), target)
            loss_vanilla_train_final.append(losst.item())
            losst.backward()
            optimizer.step()

    plt.plot(dnn_vanilla_normal_post.train_loss_seq)