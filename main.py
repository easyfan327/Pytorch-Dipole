import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pickle as pickle
import random
import logging
from sklearn.metrics import roc_auc_score


def init_params(params: dict):
    params["n_epoches"] = 100
    params["batch_size"] = 100
    params["test_ratio"] = 0.2
    params["validation_ratio"] = 0.1
    params["sequence_file"] = "retaindataset.3digitICD9.seqs"
    params["label_file"] = "retaindataset.morts"

    params["num_embeddings"] = 942
    params["embedding_dim"] = 128

    params["rnn_input_size"] = 128
    params["rnn_hidden_size"] = 128
    params["is_rnn_bidirectional"] = True

    params["attention_type"] = "location"
    # "concat"
    # "general"
    # "location"
    if params["is_rnn_bidirectional"]:
        params["attn_input_dim"] = 2 * params["rnn_hidden_size"]
        params["latent_dim"] = 128
        params["r"] = 64
    else:
        params["attn_input_dim"] = params["rnn_hidden_size"]
        params["latent_dim"] = 128
        params["r"] = 64

    params["num_classes"] = 2


class DipoleRNN(nn.Module):
    def __init__(self, params: dict):
        super(DipoleRNN, self).__init__()

        self.emb_layer = nn.Linear(in_features=params["num_embeddings"],
                                   out_features=params["embedding_dim"])

        self.emb_relu = nn.ReLU()

        self.rnn = nn.GRU(input_size=params["rnn_input_size"],
                          hidden_size=params["rnn_hidden_size"],
                          bidirectional=params["is_rnn_bidirectional"])

        if params["attention_type"] == "location":
            self.attn_layer = nn.Linear(in_features=params["attn_input_dim"],
                                        out_features=1)
            self.attn_layer_post = nn.Softmax(dim=0)
            self.attn_type = "location"
        elif params["attention_type"] == "general":
            self.attn_layer = nn.Bilinear(in1_features=params["attn_input_dim"],
                                          in2_features=params["attn_input_dim"],
                                          out_features=1,
                                          bias=False)
            self.attn_type = "general"
        elif params["attention_type"] == "concat":
            self.attn_layer = nn.Linear(in_features=params["attn_input_dim"] * 2,
                                        out_features=params["latent_dim"])
            self.attn_layer_post = nn.Linear(in_features=params["latent_dim"],
                                             out_features=1)
            self.attn_type = "concat"
            pass

        self.hidden_tilde_layer = nn.Linear(in_features=params["rnn_hidden_size"] * 2,
                                            out_features=params["r"],
                                            bias=False)
        self.output_layer = nn.Linear(in_features=params["r"],
                                      out_features=params["num_classes"])

        self.param_dict = params

    def forward(self, input, hidden):
        logging.debug("input:" + str(input.shape))
        v = self.emb_layer(input)
        v = self.emb_relu(v)
        logging.debug("v:" + str(v.shape))

        gru_output, hidden = self.rnn(v, hidden)
        logging.debug("GRU hidden:" + str(hidden.shape))
        logging.debug("GRU output:" + str(gru_output.shape))

        """
        # embedding_dim = hidden_size
        # 1 for 1 layer
        # 0 for forwarding
        # view(seq_len, batch_size, num_directions, hidden_size)
        h_f = gru_output.view(gru_output.shape[0], gru_output.shape[1], 0, gru_output.shape[2])
        h_b = gru_output.view(gru_output.shape[0], gru_output.shape[1], 1, gru_output.shape[2])

        logging.debug("h_f:" + str(h_f.shape))
        logging.debug("h_b:" + str(h_b.shape))
        h_i = torch.cat(h_f, h_b, dim=-1).squeeze(0)
        logging.debug("h:" + str(h_i.shape))
        """
        h_i = gru_output

        if self.attn_type == "location":
            alpha = self.attn_layer(h_i)
            alpha = self.attn_layer_post(alpha)
        elif self.attn_type == "general":
            h_t = h_i.unsqueeze(0).repeat(h_i.shape[0], 1, 1)
            alpha = self.attn_layer(h_t, h_i)
        elif self.attn_type == "concat":
            h_t = h_i.unsqueeze(0).repeat(h_i.shape[0], 1, 1)
            h_c = torch.cat(h_i, h_t)
            h_c = torch.tanh(self.attn_layer(h_c))
            alpha = self.attn_layer_post(h_c)
        else:
            alpha = torch.ones_like(h_i)

        logging.debug("h_i:" + str(h_i.shape))
        logging.debug("alpha:" + str(alpha.shape))
        c = alpha * h_i
        logging.debug("c:" + str(c.shape))

        # features is on the last dimension
        h_concat = torch.cat((c, h_i), dim=-1)
        logging.debug("h_concat:" + str(h_concat.shape))

        h_tilde = self.hidden_tilde_layer(h_concat)
        logging.debug("h_tilde:" + str(h_concat.shape))

        output = self.output_layer(h_tilde)
        logging.debug("output:" + str(output.shape))
        output = F.softmax(output, dim=0)
        logging.debug("output:" + str(output.shape))

        return output

    def init_hidden(self, current_batch_size):
        if self.param_dict["is_rnn_bidirectional"]:
            return torch.zeros(2, current_batch_size, self.param_dict["rnn_hidden_size"]).to(device)
        else:
            return torch.zeros(1, current_batch_size, self.param_dict["rnn_hidden_size"]).to(device)


def padMatrixWithoutTime(seqs, options):
    lengths = np.array([len(seq) for seq in seqs]).astype('int32')
    n_samples = len(seqs)
    maxlen = np.max(lengths)

    x = np.zeros((maxlen, n_samples, options['num_embeddings']))
    for idx, seq in enumerate(seqs):
        for xvec, subseq in zip(x[:, idx, :], seq):
            xvec[subseq] = 1.

    return x, lengths

def init_data(params: dict):
    sequences = np.array(pickle.load(open(params["sequence_file"], 'rb')))
    labels = np.array(pickle.load(open(params["label_file"], 'rb')))

    data_size = len(labels)
    ind = np.random.permutation(data_size)

    test_size = int(params["test_ratio"] * data_size)
    validation_size = int(params["validation_ratio"] * data_size)

    test_indices = ind[:test_size]
    valid_indices = ind[test_size:test_size + validation_size]
    train_indices = ind[test_size + validation_size:]

    train_set_x = sequences[train_indices]
    train_set_y = labels[train_indices]
    test_set_x = sequences[test_indices]
    test_set_y = labels[test_indices]
    valid_set_x = sequences[valid_indices]
    valid_set_y = labels[valid_indices]

    # setx_tensor = torch.from_numpy(xbpad)
    # sety_tensor = torch.from_numpy(train_set_y)

    # train_ds = TensorDataset(torch.from_numpy(train_set_x.values), torch.from_numpy(train_set_y.values))
    # train_dl = DataLoader(train_ds, batch_size=params["batch_size"])

    def len_argsort(seq):
        return sorted(range(len(seq)), key=lambda x: len(seq[x]))

    train_sorted_index = len_argsort(train_set_x)
    train_set_x = [train_set_x[i] for i in train_sorted_index]
    train_set_y = [train_set_y[i] for i in train_sorted_index]

    valid_sorted_index = len_argsort(valid_set_x)
    valid_set_x = [valid_set_x[i] for i in valid_sorted_index]
    valid_set_y = [valid_set_y[i] for i in valid_sorted_index]

    test_sorted_index = len_argsort(test_set_x)
    test_set_x = [test_set_x[i] for i in test_sorted_index]
    test_set_y = [test_set_y[i] for i in test_sorted_index]

    return train_set_x, train_set_y, valid_set_x, valid_set_y, test_set_x, test_set_y


if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # print(device)

    parameters = dict()
    init_params(parameters)

    train_set_x, train_set_y, valid_set_x, valid_set_y, test_set_x, test_set_y = init_data(parameters)

    model = DipoleRNN(params=parameters).to(device)
    # for name, parm in model.named_parameters():
    #   print(name, parm)
    optimizer = torch.optim.Adadelta(model.parameters(), lr=1.0, rho=0.9, eps=1e-6, weight_decay=0.001)
    loss_fn = torch.nn.CrossEntropyLoss()

    n_batches = int(np.ceil(float(len(train_set_y)) / float(parameters["batch_size"])))
    best_valid_auc = 0
    best_test_auc = 0
    best_epoch = 0
    for epoch in range(parameters["n_epoches"]):
        model.train()
        loss_vector = torch.zeros(n_batches, dtype=torch.float)
        for index in random.sample(range(n_batches), n_batches):
            xb = train_set_x[index * parameters["batch_size"]:(index + 1) * parameters["batch_size"]]
            yb = train_set_y[index * parameters["batch_size"]:(index + 1) * parameters["batch_size"]]
            xbpad, xbpad_lengths = padMatrixWithoutTime(seqs=xb, options=parameters)
            xbpadtensor = torch.from_numpy(xbpad).float().to(device)
            ybtensor = torch.from_numpy(np.array(yb)).long().to(device)
            # print(xbpadtensor.shape)
            rnn_hidden_init = model.init_hidden(xbpadtensor.shape[1])

            pred, rnn_hidden_init = model(xbpadtensor, rnn_hidden_init)
            pred = pred.squeeze(1)
            # print("pred:")
            # print(pred.shape)
            # print(pred.data)
            # print("ybtensor:")
            # print(ybtensor.shape)

            loss = loss_fn(pred, ybtensor)
            loss.backward()
            loss_vector[index] = loss
            optimizer.step()
            optimizer.zero_grad()

        model.eval()
        x, x_length = padMatrixWithoutTime(seqs=valid_set_x, options=parameters)
        x = torch.from_numpy(x).float().to(device)
        y_true = torch.from_numpy(np.array(valid_set_y)).long().to(device)
        rnn_hidden_init = model.init_hidden(x.shape[1])
        y_hat, rnn_hidden_init = model(x, rnn_hidden_init)
        y_true = y_true.unsqueeze(1)
        y_true_oh = torch.zeros(y_hat.shape).to(device).scatter_(1, y_true, 1)
        auc = roc_auc_score(y_true=y_true_oh.detach().cpu().numpy(), y_score=y_hat.detach().cpu().numpy())

        x, x_length = padMatrixWithoutTime(seqs=test_set_x, options=parameters)
        x = torch.from_numpy(x).float().to(device)
        y_true = torch.from_numpy(np.array(test_set_y)).long().to(device)
        rnn_hidden_init = model.init_hidden(x.shape[1])
        y_hat, rnn_hidden_init = model(x, rnn_hidden_init)
        y_true = y_true.unsqueeze(1)
        y_true_oh = torch.zeros(y_hat.shape).to(device).scatter_(1, y_true, 1)
        test_auc = roc_auc_score(y_true=y_true_oh.detach().cpu().numpy(), y_score=y_hat.detach().cpu().numpy())

        if test_auc > best_test_auc:
            best_test_auc = test_auc
            best_epoch = epoch

        print("{},{},{},{}".format(epoch, torch.mean(loss_vector), auc, test_auc))
