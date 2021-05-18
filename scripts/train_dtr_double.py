import os
import notifyhub
import torch
import argparse
import json
import torch.optim as optim
import torch.utils.data as tdata
import math
import numpy as np
from tqdm import tqdm

from directory import CONFIG_DIR, SAVE_DIR, NOTIFYHUB_FP
from nets.dtr import DTR


DNA_BASES = ['A', 'T', 'C', 'G']


class Dataset(tdata.Dataset):
    def __init__(self, cluster_idxs: np.ndarray, reference_file: os.path, read_dir: os.path, read_batchsize: int,
                 window_size: int):
        super(Dataset, self).__init__()
        for i in cluster_idxs:
            read_file = os.path.join(read_dir, '{}.txt'.format(i))
            assert os.path.exists(read_file)
        with open(reference_file, 'r') as f:
            references = f.readlines()
        references = [reference.strip() for reference in references]
        references = [references[i] for i in cluster_idxs]
        references = [self.to_tensor(strand=line) for line in references]
        self.references = [torch.argmax(line, dim=1).reshape(-1) for line in references]

        self.read_batchsize = read_batchsize
        self.cluster_idxs = cluster_idxs
        self.window_size = window_size
        self.read_dir = read_dir

    @staticmethod
    def to_tensor(strand: str) -> torch.Tensor:
        strand = strand.upper()
        strand = np.array(list(strand))
        out = torch.zeros(size=[strand.shape[0], 5], dtype=torch.float)
        for i, base in enumerate(DNA_BASES):
            idxs = np.argwhere(strand == base).reshape(-1)
            out[idxs, i] = 1
        return out

    def __getitem__(self, i):
        reference = self.references[i]
        cluster_id = self.cluster_idxs[i]
        read_file = os.path.join(self.read_dir, '{}.txt'.format(cluster_id))
        with open(read_file, 'r') as f:
            reads = f.readlines()

        reads = [read.strip() for read in reads]
        sample_idxs = np.random.choice(len(reads), size=self.read_batchsize, replace=True)
        reads = [reads[i] for i in sample_idxs]
        reads = [self.to_tensor(read) for read in reads]

        out_length = reference.shape[0]
        forward_len = int(math.ceil(out_length // 2)) + self.window_size
        backward_len = out_length // 2 + self.window_size

        forward_reads = torch.zeros(size=[self.read_batchsize, forward_len, 5], dtype=torch.float)
        backward_reads = torch.zeros(size=[self.read_batchsize, backward_len, 5], dtype=torch.float)
        for i, read in enumerate(reads):
            read_len = read.shape[0]  # Ti x 5

            fill_len = min(read_len, forward_len)
            forward_reads[i, :fill_len] = read[:fill_len]
            forward_reads[i, fill_len:, -1] = 1  # last characters are null characters.

            read = torch.flip(read, dims=[0])
            fill_len = min(read_len, backward_len)
            backward_reads[i, :fill_len] = read[:fill_len]
            backward_reads[i, fill_len:, -1] = 1
        return forward_reads, backward_reads, reference, forward_len, backward_len

    def collate_fn(self, batch):
        forward_reads, backward_reads, references, forward_lens, backward_lens = zip(*batch)
        batchsize = len(forward_reads)
        max_forward_len = int(max(forward_lens))
        max_backward_len = int(max(backward_lens))

        # process reads.
        def process_batch_reads(_batch_reads: list, _max_len: int) -> torch.Tensor:
            _out = torch.zeros(size=[batchsize, self.read_batchsize, _max_len, 5], dtype=torch.float)  # B x R x T x 5
            for _i, _reads in enumerate(_batch_reads):
                _read_len = _reads.shape[1]
                _out[_i, :, :_read_len] = _reads
                _out[_i, :, _read_len:, -1] = 1
            return _out
        forward_reads = process_batch_reads(_batch_reads=forward_reads, _max_len=max_forward_len)
        backward_reads = process_batch_reads(_batch_reads=backward_reads, _max_len=max_backward_len)

        # process references
        max_reference_len = max([reference.shape[0] for reference in references])
        batch_references = torch.ones(size=[batchsize, max_reference_len], dtype=torch.long) * len(DNA_BASES)
        for i, reference in enumerate(references):
            reference_len = reference.shape[0]
            batch_references[i, reference_len:] = reference
        return forward_reads, backward_reads, batch_references, forward_lens, backward_lens

    def __len__(self):
        return len(self.cluster_idxs)


class Trainer:
    def __init__(self, config: str, device: torch.device, instance: str = '0000'):
        super(Trainer, self).__init__()

        experiment = '{}-{}'.format(config, instance)
        config_file = os.path.join(CONFIG_DIR, config + '.json')
        with open(config_file, 'r') as f:
            config = json.load(f)
        self.device = device

        save_dir = os.path.join(SAVE_DIR, experiment)
        self.checkpoint_dir = os.path.join(save_dir, 'checkpoints')
        self.log_dir = os.path.join(save_dir, 'logs')
        self.result_dir = os.path.join(save_dir, 'results')

        net = DTR(emb_out_ndims=config['embed']['out ndims'], emb_base_ndims=config['embed']['base ndims'],
                  emb_nlayers=config['embed']['nlayers'],
                  window_out_ndims=config['window']['out ndims'], window_base_ndims=config['embed']['base ndims'],
                  window_size=config['window']['window size'],
                  lstm_out_ndims=config['lstm']['out ndims'],
                  update_base_ndims=config['update']['base ndims'],
                  global_natt_heads=config['global']['natt heads'],
                  pred_base_ndims=config['pred']['base ndims'],
                  activation=config['activation'], batchnorm=config['batchnorm'])
        self.net = net.to(self.device)

        optimizer_config = config['optimizer']
        optimizer = optimizer_config['name']
        if optimizer == 'adam':
            self.optimizer = optim.Adam(lr=optimizer_config['lr'], params=net.parameters())
        else:
            raise ValueError('no such optimizer')

        self.max_nepochs = config['max nepochs']
        self.train_batchsize = config['train batchsize']
        self.test_batchsize = config['test batchsize']
        self.epoch = 0

        self.loss_fn = torch.nn.CrossEntropyLoss()

    def train(self):
        if self.epoch == self.max_nepochs:
            print('INFO: training completed')
            return

        start_epoch = self.epoch
        for i in range(start_epoch, self.max_nepochs):
            # self.train_epoch(dataset=)
            pass

    def train_epoch(self, dataset: Dataset) -> float:
        self.net.train()
        dataloader = tdata.DataLoader(dataset=dataset, batch_size=self.train_batchsize, collate_fn=dataset.collate_fn,
                                      num_workers=6, shuffle=True, drop_last=True)

        losses = []
        pbar = tqdm(dataloader)
        for forward_reads, backward_reads, forward_lens, backward_lens, references in pbar:
            batchsize = len(forward_reads)
            predictions = self.get_predictions(forward_reads=forward_reads, backward_reads=backward_reads,
                                               forward_lens=forward_lens, backward_lens=backward_lens)
            references = references.to(self.device)

            loss = torch.tensor(0., device=self.device)
            for i, prediction in enumerate(predictions):
                reference = references[i]
                loss += self.loss_fn(reference, loss)
            loss /= batchsize

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            losses.append(loss.item())
        avg_loss = float(np.mean(losses).item())
        return avg_loss



    def get_predictions(self, forward_reads: torch.Tensor, backward_reads: torch.Tensor,
                        forward_lens: list, backward_lens: list) -> list:
        batchsize, nreads, _, _ = forward_reads.shape
        max_read_len = max([forward_reads.shape[2], backward_reads.shape[2]])
        reads = torch.zeros(size=[batchsize * 2, nreads, max_read_len, 5])
        reads[:, :, :, -1] = 1
        reads[:batchsize, :forward_reads.shape[2]] = forward_reads
        reads[batchsize:, :backward_reads.shape[2]] = backward_reads
        reads = reads.to(self.device)

        out = self.net(reads, max_read_len)
        predictions = []
        for i, forward_len in enumerate(forward_lens):
            backward_len = backward_lens[i]
            forwad_prediction = out[i, :forward_len]
            backward_prediction = torch.flip(out[i, :backward_len], dims=[0])
            prediction = torch.cat([forwad_prediction, backward_prediction], dim=0)
            predictions.append(prediction)
        return predictions


@notifyhub.watch(config_fp=NOTIFYHUB_FP)
def main():
    argparser = argparse.ArgumentParser()
    argparser.add_argument('--device', type=int, choices=list(range(torch.cuda.device_count())), default=0)
    argparser.add_argument('--config', type=str, required=True)
    args = argparser.parse_args()


if __name__ == '__main__':
    main()
