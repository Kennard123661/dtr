import torch.nn as nn
import torch
from nets import Linear, MlpNet, Conv1dNet


class PredictionNet(nn.Module):
    def __init__(self, in_ndims: int, out_ndims: int, base_ndims: int, nlayers: int, activation: str = None,
                 normalization: str = None):
        super(PredictionNet, self).__init__()

        if nlayers == 2:
            net = [Linear(in_ndims=in_ndims, out_ndims=base_ndims, activation=activation, normalization=normalization)]
        elif nlayers > 2:
            net = [MlpNet(in_ndims=in_ndims, out_ndims=base_ndims, base_ndims=base_ndims, activation=activation,
                          nlayers=nlayers-1, normalization=normalization)]
        else:
            net = list()

        if nlayers >= 2:
            in_ndims = base_ndims
        net.append(Linear(in_ndims=in_ndims, out_ndims=out_ndims, activation=None, normalization=normalization))
        self.net = nn.Sequential(*net)

    def forward(self, x: torch.Tensor):
        out = self.net(x)
        return out


class WindowEmbeddingNet(nn.Module):
    def __init__(self, in_ndims, out_ndims: int, base_ndims: int, window_size: int, activation: str = None,
                 normalization: str = None):
        super(WindowEmbeddingNet, self).__init__()
        self.in_ndims = in_ndims
        self.window_size = window_size
        self.net = MlpNet(in_ndims=in_ndims * window_size, out_ndims=out_ndims, base_ndims=base_ndims, nlayers=2,
                          activation=activation, normalization=normalization)

    def forward(self, x: torch.Tensor):
        assert len(x.shape) == 3, 'BR x D x W'
        nstrands, ndims, window_size = x.shape
        assert self.in_ndims * self.window_size == window_size * ndims
        out = x.view(nstrands, ndims * window_size)
        out = self.net(out)
        return out


class Lstm(nn.Module):
    def __init__(self, in_ndims: int, out_ndims: int):
        super(Lstm, self).__init__()
        self.lstm = nn.LSTMCell(input_size=in_ndims, hidden_size=out_ndims)
        self.out_ndims = out_ndims

    def get_initial_states(self, batchsize: int, device) -> (torch.Tensor, torch.Tensor):
        c_states = torch.zeros([batchsize, self.out_ndims], device=device)
        h_states = torch.zeros([batchsize, self.out_ndims], device=device)
        return c_states, h_states

    def forward(self, window_in: torch.Tensor, c_states: torch.Tensor, h_states: torch.Tensor):
        h_states, c_states = self.lstm(window_in, (h_states, c_states))
        return c_states, h_states


class MultiHeadAttentionAggregation(nn.Module):
    def __init__(self, in_ndims: int, out_ndims: int, att_ndims: int, natt_heads: int, activation: str = None,
                 normalization: str = None):
        super(MultiHeadAttentionAggregation, self).__init__()
        self.att_nets = nn.ModuleList([SelfAttentionAggregation(in_ndims=in_ndims, out_ndims=out_ndims,
                                                                att_ndims=att_ndims) for _ in range(natt_heads)])
        self.net = Linear(in_ndims=out_ndims * natt_heads, out_ndims=out_ndims, activation=activation,
                          normalization=normalization)

    def forward(self, batch_reads: torch.Tensor):
        out = [att_net(batch_reads) for att_net in self.att_nets]
        out = torch.cat(out, dim=1)  # B x AD
        out = self.net(out)
        return out


class SelfAttentionAggregation(nn.Module):
    def __init__(self, in_ndims: int, out_ndims: int, att_ndims: int):
        """

        Args:
            in_ndims: input dimensions
            out_ndims: output_dimensions
            att_ndims:
        """
        super(SelfAttentionAggregation, self).__init__()
        self.net_q = nn.Linear(in_features=in_ndims, out_features=att_ndims)
        self.net_k = nn.Linear(in_features=in_ndims, out_features=att_ndims)
        self.net_v = nn.Linear(in_features=in_ndims, out_features=out_ndims)

    def forward(self, batch_reads: torch.Tensor):
        batchsize, nreads, ndims = batch_reads.shape
        reads = batch_reads.view(batchsize * nreads, ndims)

        query_feats = self.net_q(reads).view(batchsize, nreads, -1)  # B x R x D
        key_feats = self.net_k(reads).view(batchsize, nreads, -1)  # B x R x D
        value_feats = self.net_v(reads).view(batchsize, nreads, -1)  # B x R x D

        att_matrix = torch.bmm(query_feats, key_feats.permute(0, 2, 1))  # B x R x R
        att_matrix = torch.softmax(att_matrix, dim=2)

        out = torch.bmm(att_matrix, value_feats)  # B x R x D
        out = torch.mean(out, dim=1)  # B x D
        return out


class DTR(nn.Module):
    def __init__(self, emb_out_ndims: int, emb_base_ndims: int, emb_nlayers: int,
                 window_size: int, window_out_ndims, window_base_ndims: int,
                 lstm_out_ndims: int,
                 update_base_ndims: int,
                 global_natt_heads: int,
                 pred_base_ndims: int,
                 activation: str = None, normalization: str = None):
        super(DTR, self).__init__()
        self.embedding_net = Conv1dNet(in_ndims=4, out_ndims=emb_out_ndims, base_ndims=emb_base_ndims, ksize=3,
                                       padding=1, nlayers=emb_nlayers, activation=activation,
                                       normalization=normalization)
        self.belief_initializer = Conv1dNet(in_ndims=4, out_ndims=emb_out_ndims, base_ndims=emb_base_ndims, ksize=3,
                                            padding=1, nlayers=emb_nlayers, activation=activation,
                                            normalization=normalization)
        self.belief_normalization = nn.InstanceNorm1d(emb_out_ndims)

        self.window_size = window_size
        self.window_net = WindowEmbeddingNet(in_ndims=emb_out_ndims*2, out_ndims=window_out_ndims,
                                             base_ndims=window_base_ndims,
                                             window_size=window_size, activation=activation,
                                             normalization=normalization)

        self.lstm = Lstm(in_ndims=window_out_ndims, out_ndims=lstm_out_ndims)

        self.global_feat_net = MultiHeadAttentionAggregation(in_ndims=lstm_out_ndims, out_ndims=lstm_out_ndims,
                                                             att_ndims=lstm_out_ndims, natt_heads=global_natt_heads,
                                                             activation=activation, normalization=normalization)

        self.update_net = MlpNet(in_ndims=lstm_out_ndims * 2, out_ndims=emb_out_ndims*(window_size-1),
                                 base_ndims=update_base_ndims, nlayers=2, activation=activation,
                                 normalization=normalization)

        self.prediction_net = MlpNet(in_ndims=lstm_out_ndims, out_ndims=5, nlayers=2, base_ndims=pred_base_ndims,
                                     activation=activation, normalization=normalization)

    def forward(self, reads: torch.Tensor, out_length: int):
        batchsize, nreads, sequence_length, _ = reads.shape
        nstrands = batchsize * nreads
        reads = reads.view(nstrands, sequence_length, 4)
        reads = reads.permute(0, 2, 1)  # BR x 5 x L

        original = self.embedding_net(reads)  # BR x D x L
        belief = self.belief_initializer(reads)  # BR x D x L

        c_states, h_states = self.lstm.get_initial_states(batchsize=nstrands, device=reads.device)
        assert int(out_length) <= int(sequence_length + self.window_size)
        predictions = []
        for start in range(out_length):
            end = start + self.window_size

            original_window = original[:, :, start:end]  # BR x D x W
            belief_window = belief[:, :, start:end]  # BR x D x W
            belief_window = self.belief_normalization(belief_window)

            window_inputs = torch.cat([belief_window, original_window], dim=1)  # BR x 2D x W
            window_inputs = self.window_net(window_inputs)  # BR x DW

            c_states, h_states = self.lstm(window_inputs, c_states, h_states)  # BR x D
            lstm_out = h_states.view(batchsize, nreads, h_states.shape[-1])
            global_features = self.global_feat_net(lstm_out)  # B x D

            # update the belief
            update_in = torch.cat([lstm_out, global_features.view(batchsize, 1, -1).expand(-1, nreads, -1)], dim=2)
            update_in = update_in.view(nstrands, -1)  # BR x D
            update_out = self.update_net(update_in)  # BR x D(W-1)
            update_out = update_out.view(nstrands, -1, self.window_size - 1)  # BR x D x (W-1)
            belief[:, :, start+1:end] += update_out

            # get the predictions
            pred_out = self.prediction_net(global_features)  # B x 5
            predictions.append(pred_out)
        predictions = torch.stack(predictions, dim=1)  # B x L x 5
        assert predictions.shape == (batchsize, out_length, 5)
        return predictions


def main():
    import json
    config_file = '/home/kennardng/projects/dtr/configs/base-instance.json'
    # config_file = '/home/kennardngpoolhua/Desktop/deep-trace/configs/lstm/base-atth4-er10-w14.json'
    with open(config_file, 'r') as f:
        config = json.load(f)
    config = config['net']
    out_length = 120

    model = DTR(emb_out_ndims=config['embed']['out ndims'], emb_base_ndims=config['embed']['base ndims'],
                emb_nlayers=config['embed']['nlayers'],
                window_out_ndims=config['window']['out ndims'], window_base_ndims=config['embed']['base ndims'],
                window_size=config['window']['window size'],
                lstm_out_ndims=config['lstm']['out ndims'],
                update_base_ndims=config['update']['base ndims'],
                global_natt_heads=config['global']['natt heads'],
                pred_base_ndims=config['pred']['base ndims'],
                activation=config['activation'], normalization=config['normalization'])
    forward_data = torch.randn(size=[3, 10, 130, 4]).float()
    backward_data = torch.randn(size=[3, 10, 130, 4]).float()
    out = model(forward_data, out_length)
    print(out.shape)


if __name__ == '__main__':
    main()
