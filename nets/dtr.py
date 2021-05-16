import torch.nn as nn
import torch
from nets import Conv1d, Linear, get_activation


# torch.autograd.set_detect_anomaly(True)
class MlpNet(nn.Module):
    def __init__(self, config: dict):
        super(MlpNet, self).__init__()
        in_dims = config['in-dims']
        out_dims = config['out-dims']

        self.out_dims = out_dims

        base_dims = config['base-dims']
        num_layers = config['num-layers']
        activation = config['activation']
        has_bn = config['batch-norm']

        net = []
        net.append(Linear(in_dims=in_dims, out_dims=base_dims, activation=activation, batchnorm=has_bn))
        for _ in range(num_layers - 2):
            net.append(Linear(in_dims=base_dims, out_dims=base_dims, activation=activation, batchnorm=has_bn))
        net.append(Linear(in_dims=base_dims, out_dims=out_dims, activation=activation, batchnorm=has_bn))
        self.net = nn.Sequential(*net)

    def forward(self, reads: torch.Tensor):
        out = self.net(reads)
        return out


class PredictionNet(nn.Module):
    def __init__(self, config: dict):
        super(PredictionNet, self).__init__()
        in_dims = config['in-dims']
        out_dims = config['out-dims']
        base_dims = config['base-dims']
        num_layers = config['num-layers']
        activation = config['activation']
        has_bn = config['batch-norm']

        net = []
        net.append(Linear(in_dims=in_dims, out_dims=base_dims, activation=activation, batchnorm=has_bn))
        for _ in range(num_layers - 2):
            net.append(Linear(in_dims=base_dims, out_dims=base_dims, activation=activation, batchnorm=has_bn))
        net.append(Linear(in_dims=base_dims, out_dims=out_dims, activation=None, batchnorm=has_bn))
        self.net = nn.Sequential(*net)

    def forward(self, reads: torch.Tensor):
        out = self.net(reads)
        return out


class MultiConvNet(nn.Module):
    def __init__(self, config: dict):
        super(MultiConvNet, self).__init__()
        in_dims = config['in-dims']
        out_dims = config['out-dims']
        self.out_dims = out_dims

        base_dims = config['base-dims']
        num_layers = config['num-layers']
        activation = config['activation']
        has_bn = config['batch-norm']

        net = []
        net.append(Conv1d(in_channels=in_dims, out_channels=base_dims, kernel_size=3,
                          padding=1, activation=activation, batchnorm=has_bn))
        for _ in range(num_layers - 2):
            net.append(Conv1d(in_channels=base_dims, out_channels=base_dims, kernel_size=3,
                              padding=1, activation=activation, batchnorm=has_bn))
        net.append(Conv1d(in_channels=base_dims, out_channels=self.out_dims, kernel_size=3,
                          padding=1, activation=activation, batchnorm=has_bn))
        self.net = nn.Sequential(*net)

    def forward(self, reads: torch.Tensor):
        assert len(reads.shape) == 3
        out = self.net(reads)
        return out


class WindowEmbeddingNet(nn.Module):
    def __init__(self, config: dict):
        super(WindowEmbeddingNet, self).__init__()
        in_dims = config['in-dims']
        window_size = config['window-size']
        out_dims = config['out-dims']
        base_dims = config['base-dims']
        activation = config['activation']
        has_bn = config['batch-norm']

        self.net = MlpNet(config={'in-dims': in_dims * window_size, 'out-dims': out_dims, 'num-layers': 2,
                                  'base-dims': base_dims,
                                  'activation': activation, 'batch-norm': has_bn})

    def forward(self, inputs: torch.Tensor):
        assert len(inputs.shape) == 3, 'BR x D x W'
        num_strands = inputs.shape[0]
        out = inputs.view(num_strands, -1)
        out = self.net(out)
        return out


class LstmModule(nn.Module):
    def __init__(self, config: dict):
        super(LstmModule, self).__init__()
        in_dims = config['in-dims']
        out_dims = config['out-dims']
        self.out_dims = out_dims
        self.lstm = nn.LSTMCell(input_size=in_dims, hidden_size=out_dims)

    def get_initial_states(self, batchsize: int, device) -> (torch.Tensor, torch.Tensor):
        c_states = torch.zeros([batchsize, self.out_dims], device=device)
        h_states = torch.zeros([batchsize, self.out_dims], device=device)
        return c_states, h_states

    def forward(self, window_in: torch.Tensor, c_states: torch.Tensor, h_states: torch.Tensor):
        h_states, c_states = self.lstm(window_in, (h_states, c_states))
        return c_states, h_states


class MultiHeadAttentionAggregation(nn.Module):
    def __init__(self, config: dict):
        super(MultiHeadAttentionAggregation, self).__init__()
        num_heads = config['num-heads']

        self.att_nets = nn.ModuleList([SelfAttentionAggregation(config=config) for _ in range(num_heads)])
        out_dims = config['out-dims']
        in_dims = out_dims * num_heads
        activation = config['activation']
        has_bn = config['batch-norm']
        self.net = Linear(in_dims=in_dims, out_dims=out_dims, activation=activation, batchnorm=has_bn)

    def forward(self, batch_reads: torch.Tensor):
        out_features = [att_net(batch_reads) for att_net in self.att_nets]
        out_features = torch.cat(out_features, dim=1)
        out_features = self.net(out_features)
        return out_features


class SelfAttentionAggregation(nn.Module):
    def __init__(self, config: dict):
        super(SelfAttentionAggregation, self).__init__()
        in_dims = config['in-dims']
        out_dims = config['out-dims']
        att_dims = config['att-dims']
        if 'activation' in config:
            activation = config['activation']
        else:
            activation = None
        self.net_q = nn.Linear(in_features=in_dims, out_features=att_dims)
        self.net_k = nn.Linear(in_features=in_dims, out_features=att_dims)
        self.net_v = nn.Linear(in_features=in_dims, out_features=out_dims)
        self.act = get_activation(activation=activation)

    def forward(self, batch_reads: torch.Tensor):
        batchsize, num_reads, num_dims = batch_reads.shape
        reads = batch_reads.view(batchsize * num_reads, num_dims)

        query_feats = self.net_q(reads).view(batchsize, num_reads, -1)  # B x R x D
        key_feats = self.net_k(reads).view(batchsize, num_reads, -1)  # B x R x D
        value_feats = self.net_v(reads).view(batchsize, num_reads, -1)  # B x R x D

        att_matrix = torch.bmm(query_feats, key_feats.permute(0, 2, 1))  # B x R x R
        att_matrix = torch.softmax(att_matrix, dim=2)

        out_features = torch.bmm(att_matrix, value_feats)  # B x R x D
        out_features = torch.mean(out_features, dim=1)
        out_features = self.act(out_features)  # B x D
        return out_features


class LstmModel(nn.Module):
    def __init__(self, out_length: int, config: dict):
        super(LstmModel, self).__init__()
        self.out_length = out_length

        embedding_config = config['embedding']
        assert embedding_config['in-dims'] == 5
        self.embedding_net = MultiConvNet(config=embedding_config)
        self.belief_initializer = MultiConvNet(config=embedding_config)

        window_config = config['window']
        window_size = window_config['window-size']
        window_dims = self.embedding_net.out_dims * window_size
        window_config['in-dims'] = self.embedding_net.out_dims + self.belief_initializer.out_dims
        self.window_net = WindowEmbeddingNet(config=window_config)
        self.window_size = window_size

        self.normalize_belief = config['normalize-belief']
        self.normalize_hidden = config['normalize-hidden']
        lstm_config = config['lstm']
        self.lstm = LstmModule(config=lstm_config)
        self.belief_instance_norm = nn.InstanceNorm1d(self.embedding_net.out_dims)
        self.hidden_instance_norm = nn.InstanceNorm1d(self.lstm.out_dims)
        self.global_feat_net = MultiHeadAttentionAggregation(config={'in-dims': self.lstm.out_dims,
                                                                     'out-dims': self.lstm.out_dims,
                                                                     'att-dims': self.lstm.out_dims,
                                                                     'num-heads': lstm_config['num-att-heads'],
                                                                     'activation': lstm_config['activation'],
                                                                     'batch-norm': lstm_config['batch-norm']})

        self.update_out_dims = self.embedding_net.out_dims * (window_size - 1)
        self.update_net = MlpNet(config={'in-dims': self.lstm.out_dims * 2,
                                         'out-dims': self.update_out_dims,
                                         'base-dims': lstm_config['base-dims'],
                                         'num-layers': 2, 'activation': lstm_config['activation'],
                                         'batch-norm': lstm_config['batch-norm']})
        self.prediction_net = MlpNet(config={'in-dims': self.lstm.out_dims, 'out-dims': 4, 'num-layers': 2,
                                             "base-dims": lstm_config['base-dims'],
                                             'activation': lstm_config['activation'],
                                             'batch-norm': lstm_config['batch-norm']})

    def get_aggregated_vector(self, reads: torch.Tensor):
        batchsize, num_reads, sequence_length, _ = reads.shape
        num_strands = batchsize * num_reads
        reads = reads.view(num_strands, sequence_length, 5)
        reads = reads.permute(0, 2, 1)  # BR x 5 x L

        original = self.embedding_net(reads)  # BR x D x L
        belief = self.belief_initializer(reads)  # BR x D x L

        c_states, h_states = self.lstm.get_initial_states(batchsize=num_strands, device=reads.device)
        assert self.out_length < sequence_length + self.window_size

        aggregated_vectors = []
        for start in range(self.out_length):
            end = start + self.window_size

            original_window = original[:, :, start:end]  # BR x D x W
            belief_window = belief[:, :, start:end]  # BR x D x W
            if self.normalize_belief:
                belief_window = self.belief_instance_norm(belief_window)
            window_inputs = torch.cat([belief_window, original_window], dim=1)  # BR x D x W
            window_inputs = self.window_net(window_inputs)  # BR x DW

            c_states, h_states = self.lstm(window_inputs, c_states, h_states)  # BR x D
            if self.normalize_hidden:
                h_states = h_states.view(batchsize, num_reads, -1)  # B x R x D
                h_states = h_states.permute(0, 2, 1)  # B x D x R
                h_states = self.hidden_instance_norm(h_states)  # B x D x R
                h_states = h_states.permute(0, 2, 1)  # B x R x D
                h_states = h_states.reshape(num_strands, -1)  # BR x D

            lstm_out = h_states.view(batchsize, num_reads, -1)
            global_features = self.global_feat_net(lstm_out)  # B x D

            # update the belief
            update_in = torch.cat([lstm_out, global_features.view(batchsize, 1, -1).expand(-1, num_reads, -1)], dim=2)
            update_in = update_in.view(batchsize * num_reads, -1)  # BR x D
            update_out = self.update_net(update_in)  # BR x D(W-1)
            update_out = update_out.view(num_strands, -1, self.window_size - 1)  # BR x D x (W-1)
            belief[:, :, start + 1:end] += update_out

            aggregated_vectors.append(global_features)
        aggregated_vectors = torch.stack(aggregated_vectors, dim=1)  # B x L x D
        return aggregated_vectors

    def forward(self, reads: torch.Tensor):
        batchsize, num_reads, sequence_length, _ = reads.shape
        num_strands = batchsize * num_reads
        reads = reads.view(num_strands, sequence_length, 5)
        reads = reads.permute(0, 2, 1)  # BR x 5 x L

        original = self.embedding_net(reads)  # BR x D x L
        belief = self.belief_initializer(reads)  # BR x D x L

        c_states, h_states = self.lstm.get_initial_states(batchsize=num_strands, device=reads.device)
        assert self.out_length < sequence_length + self.window_size
        predictions = []
        for start in range(self.out_length):
            end = start + self.window_size

            original_window = original[:, :, start:end]  # BR x D x W
            belief_window = belief[:, :, start:end]  # BR x D x W
            if self.normalize_belief:
                # belief_window_norm = F.norm(belief_window, p=2, dim=1, keepdim=True)  # normalizes the belief vector
                # belief_window = belief_window.norm(, belief_window_norm)
                belief_window = self.belief_instance_norm(belief_window)
            window_inputs = torch.cat([belief_window, original_window], dim=1)  # BR x D x W
            window_inputs = self.window_net(window_inputs)  # BR x DW

            c_states, h_states = self.lstm(window_inputs, c_states, h_states)  # BR x D
            if self.normalize_hidden:
                h_states = h_states.view(batchsize, num_reads, -1)  # B x R x D
                h_states = h_states.permute(0, 2, 1)  # B x D x R
                h_states = self.hidden_instance_norm(h_states)  # B x D x R
                h_states = h_states.permute(0, 2, 1)  # B x R x D
                h_states = h_states.reshape(num_strands, -1)  # BR x D

            lstm_out = h_states.view(batchsize, num_reads, -1)
            global_features = self.global_feat_net(lstm_out)  # B x D

            # update the belief
            update_in = torch.cat([lstm_out, global_features.view(batchsize, 1, -1).expand(-1, num_reads, -1)], dim=2)
            update_in = update_in.view(batchsize * num_reads, -1)  # BR x D
            update_out = self.update_net(update_in)  # BR x D(W-1)
            update_out = update_out.view(num_strands, -1, self.window_size - 1)  # BR x D x (W-1)
            belief[:, :, start + 1:end] += update_out

            # get the predictions
            pred_out = self.prediction_net(global_features)  # B x 4
            predictions.append(pred_out)
        predictions = torch.stack(predictions, dim=1)  # B x L x 4
        assert predictions.shape[1] == self.out_length and predictions.shape[2] == 4
        return predictions


class Model(nn.Module):
    def __init__(self, output_length, config: dict):
        super(Model, self).__init__()
        self.output_length = output_length
        assert self.half_length * 2 == self.output_length
        self.net = LstmModel(out_length=self.half_length, config=config)

    @property
    def half_length(self):
        return self.output_length // 2

    def extract_global_vecs(self, forward_reads: torch.Tensor, backward_reads: torch.Tensor):
        assert forward_reads.shape == backward_reads.shape
        batchsize = forward_reads.shape[0]
        reads = torch.cat([forward_reads, backward_reads], dim=0)

        half = self.half_length
        self.net.out_length = half
        out = self.net.get_aggregated_vector(reads)  # 2B x T x D

        forward_out = out[:batchsize, :half, :]
        backward_out = out[batchsize:, :half, :]
        backward_out = torch.flip(backward_out, dims=[1])
        out = torch.cat([forward_out, backward_out], dim=1)
        return out  # B x 2T x D

    def forward(self, forward_reads: torch.Tensor, backward_reads: torch.Tensor):
        assert forward_reads.shape == backward_reads.shape
        batchsize = forward_reads.shape[0]
        reads = torch.cat([forward_reads, backward_reads], dim=0)

        half = self.half_length
        self.net.out_length = half
        out = self.net(reads)  # 2B x T x 4

        forward_out = out[:batchsize, :half, :]
        backward_out = out[batchsize:, :half, :]
        backward_out = torch.flip(backward_out, dims=[1])
        out = torch.cat([forward_out, backward_out], dim=1)
        return out


class SingleModel(Model):
    def __init__(self, output_length, config: dict):
        super(SingleModel, self).__init__(output_length=output_length, config=config)
        self.net = LstmModel(out_length=self.output_length, config=config)

    def forward(self, forward_reads: torch.Tensor, backward_reads: torch.Tensor = None):
        read_length = forward_reads.shape[2]
        self.net.out_length = self.output_length
        assert read_length >= (self.output_length + self.net.window_size)
        out = self.net(forward_reads)
        return out


def main():
    import json
    config_file = '/home/kennardng/Desktop/deep-trace/configs/lstm/base-atth4-er10-w14.json'
    # config_file = '/home/kennardngpoolhua/Desktop/deep-trace/configs/lstm/base-atth4-er10-w14.json'
    with open(config_file, 'r') as f:
        config = json.load(f)
    out_length = 120
    model = Model(output_length=out_length, config=config['model'])
    forward_data = torch.randn(size=[3, 10, 130, 5]).float()
    backward_data = torch.randn(size=[3, 10, 130, 5]).float()
    out = model(forward_data, backward_data)
    print(out.shape)


def main():
    import json
    config_file = '/home/kennardng/Desktop/deep-trace/configs/lstm/base-atth4-er10-w14.json'
    # config_file = '/home/kennardngpoolhua/Desktop/deep-trace/configs/lstm/base-atth4-er10-w14.json'
    with open(config_file, 'r') as f:
        config = json.load(f)
    out_length = 120
    model = Model(output_length=out_length, config=config['model'])
    forward_data = torch.randn(size=[3, 10, 130, 5]).float()
    backward_data = torch.randn(size=[3, 10, 130, 5]).float()
    out = model(forward_data, backward_data)
    print(out.shape)


if __name__ == '__main__':
    main()