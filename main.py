import torch
import torch.nn as nn
import numpy as np


def init_params(params: dict):
    params["batch_size"] = 100

    params["num_embeddings"] = 10
    params["embedding_dim"] = 10

    params["rnn_input_size"] = 10
    params["rnn_hidden_size"] = 10
    params["is_rnn_bidirectional"] = True


class DipoleRNN(nn.Module):
    def __init__(self, params: dict):
        super(DipoleRNN, self).__init__()

        self.emb_layer = nn.Linear(in_features=params["num_embeddings"],
                                   out_features=params["embedding_dim"])

        self.rnn = nn.GRU(input_size=params["rnn_input_size"],
                          hidden_size=params["rnn_hidden_size"],
                          bidirectional=params["is_rnn_bidirectional"])
        pass

    def forward(self, input, hidden):
        pass

    def init_hidden(self, params: dict):
        pass


if __name__ == "__main__":
    pass
