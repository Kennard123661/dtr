import os
import notifyhub
import torch
import argparse
import json
import torch.optim as optim
import torch.utils.data as tdata
import sys
import numpy as np
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import torch.multiprocessing
torch.multiprocessing.set_sharing_strategy('file_system')  # fixed shared  memory error

if __name__ == '__main__':
    PROJECT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    sys.path.insert(0, PROJECT_DIR)

from directory import CONFIG_DIR, SAVE_DIR, NOTIFYHUB_FP
from nets.dtr import DTR

DNA_BASES = ['A', 'T', 'C', 'G']
NUM_WORKERS = 12


def get_train_test_data(dataset: str) -> (np.ndarray, np.ndarray, os.path, os.path):
    if dataset == 'blast':
        import datasets.blast as dset
    else:
        raise ValueError('no such dataset')
    reference_file = dset.REFERENCE_FILE
    read_dir = dset.READ_DIR
    split_dir = dset.SPLIT_DIR
    split_file = os.path.join(split_dir, 'split-0.npy')
    test_idxs = np.load(split_file)

    with open(reference_file, 'r') as f:
        nclusters = len(f.readlines())
    train_idxs = np.setdiff1d(np.arange(nclusters), test_idxs).reshape(-1)
    return train_idxs, test_idxs, read_dir, reference_file


class Dataset(tdata.Dataset):
    def __init__(self, cluster_idxs: np.ndarray, reference_file: os.path, read_dir: os.path, window_size: int):
        super(Dataset, self).__init__()
        for i in cluster_idxs:
            read_file = os.path.join(read_dir, '{}.txt'.format(i))
            assert os.path.exists(read_file)
        with open(reference_file, 'r') as f:
            references = f.readlines()

        # parse the reference strands
        references = [reference.strip() for reference in references]
        references = [list(reference) for reference in references]
        references = [np.array([DNA_BASES.index(base) for base in reference]) for reference in references]
        references = [torch.from_numpy(reference).long() for reference in references]

        self.references = [references[i] for i in cluster_idxs]
        self.window_size = window_size

        all_reads = []
        for i in tqdm(cluster_idxs):
            read_file = os.path.join(read_dir, '{}.txt'.format(i))
            with open(read_file, 'r') as f:
                reads = f.readlines()
            reads = [read.strip() for read in reads]
            reads = [self.to_tensor(read) for read in reads]
            all_reads.append(reads)
        self.all_reads = all_reads

    @staticmethod
    def to_tensor(strand: str) -> torch.Tensor:
        strand = strand.upper()
        strand = np.array(list(strand))
        out = torch.zeros(size=[strand.shape[0], 4], dtype=torch.float)
        for i, base in enumerate(DNA_BASES):
            idxs = np.argwhere(strand == base).reshape(-1)
            out[idxs, i] = 1
        return out

    def __getitem__(self, i):
        reference = self.references[i]
        reference_length = reference.shape[0]
        reads = self.all_reads[i]

        read_lengths = [read.shape[0] for read in reads]
        batch_length = max(max(read_lengths), reference_length + self.window_size)

        # create batch sizes todo: check whether its created properly.
        num_reads = len(reads)
        forward_reads = torch.zeros(size=[num_reads, batch_length, 4], dtype=torch.float)
        backward_reads = torch.zeros(size=[num_reads, batch_length, 4], dtype=torch.float)
        for i, read in enumerate(reads):
            read_length = read_lengths[i]
            forward_reads[i, :read_length] = read

            # reverse the dna strand
            read = torch.flip(read, dims=[0])
            backward_reads[i, :read_length] = read
        return forward_reads, backward_reads, reference

    @staticmethod
    def collate_fn(batch) -> (torch.Tensor, torch.Tensor, list):
        forward_reads, backward_reads, batch_references = zip(*batch)
        assert len(forward_reads) == len(backward_reads) == len(batch_references) == 1, 'batch size should be 1'
        return forward_reads[0], backward_reads[0], batch_references[0]

    def __len__(self):
        return len(self.references)


class Trainer:
    def __init__(self, config: str, device: torch.device, instance: str = '0000'):
        super(Trainer, self).__init__()

        experiment = '{}-{}'.format(config, instance)
        config_file = os.path.join(CONFIG_DIR, 'var-num-reads', config + '.json')
        with open(config_file, 'r') as f:
            config = json.load(f)
        self.device = device
        self.config = config

        save_dir = os.path.join(SAVE_DIR, experiment)
        self.checkpoint_dir = os.path.join(save_dir, 'var-num-reads', 'checkpoints')
        self.log_dir = os.path.join(save_dir, 'logs')
        self.result_dir = os.path.join(save_dir, 'results')
        self.prediction_dir = os.path.join(self.result_dir, 'predictions')

        os.makedirs(self.log_dir, exist_ok=True)
        self.summary_writer = SummaryWriter(self.log_dir)

        self.max_nepochs = config['max nepochs']
        self.train_batchsize = config['train batchsize']
        self.epoch = 0

        net_config = config['net']
        net = DTR(emb_out_ndims=net_config['embed']['out ndims'], emb_base_ndims=net_config['embed']['base ndims'],
                  emb_nlayers=net_config['embed']['nlayers'],
                  window_out_ndims=net_config['window']['out ndims'], window_base_ndims=net_config['embed']['base ndims'],
                  window_size=net_config['window']['window size'],
                  lstm_out_ndims=net_config['lstm']['out ndims'],
                  update_base_ndims=net_config['update']['base ndims'],
                  global_natt_heads=net_config['global']['natt heads'],
                  pred_base_ndims=net_config['pred']['base ndims'],
                  activation=net_config['activation'], normalization=net_config['normalization'])
        self.net = net.to(self.device)

        optimizer_config = config['optimizer']
        optimizer = optimizer_config['name']
        if optimizer == 'adam':
            self.optimizer = optim.Adam(lr=optimizer_config['lr'] / self.train_batchsize, params=net.parameters())
        else:
            raise ValueError('no such optimizer')

        self.loss_fn = torch.nn.CrossEntropyLoss()
        self.load_checkpoint()

    def train(self):
        train_idxs, test_idxs, read_dir, reference_file = get_train_test_data(dataset=self.config['dataset'])

        if self.epoch == self.max_nepochs:
            print('INFO: training completed')
        else:
            start_epoch = self.epoch
            train_dataset = Dataset(cluster_idxs=train_idxs, reference_file=reference_file, read_dir=read_dir,
                                    window_size=self.config['net']['window']['window size'])

            for i in range(start_epoch, self.max_nepochs):
                loss = self.train_epoch(dataset=train_dataset)
                self.summary_writer.add_scalar('loss', scalar_value=loss, global_step=i)
            self.save_checkpoint()
        self.evaluate(cluster_idxs=test_idxs, reference_file=reference_file, read_dir=read_dir)

    def save_checkpoint(self, name: str = 'model'):
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        ckpt_file = os.path.join(self.checkpoint_dir, name + '.pt')
        ckpt = {
            'net': self.net.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'epoch': self.epoch
        }
        torch.save(ckpt, ckpt_file)

    def load_checkpoint(self, name: str = 'model'):
        ckpt_file = os.path.join(self.checkpoint_dir, name + '.pt')
        if os.path.exists(ckpt_file):
            ckpt = torch.load(ckpt_file, map_location='cpu')
            self.net.load_state_dict(ckpt['net'])
            self.optimizer.load_state_dict(ckpt['optimizer'])
            self.epoch = ckpt['epoch']
        else:
            print('INFO: checkpoint {} does not exist.'.format(ckpt_file))

    def train_epoch(self, dataset: Dataset) -> float:
        self.net.train()
        self.optimizer.zero_grad()
        dataloader = tdata.DataLoader(dataset=dataset, batch_size=1, collate_fn=dataset.collate_fn,
                                      num_workers=NUM_WORKERS, shuffle=True, drop_last=True)

        losses = []
        pbar = tqdm(dataloader)
        for i, (forward_reads, backward_reads, reference) in enumerate(pbar, start=1):
            reference_length = len(reference)
            reference = reference.to(self.device)

            prediction = self.get_prediction(forward_reads=forward_reads, backward_reads=backward_reads,
                                             prediction_length=reference_length)
            loss = self.loss_fn(prediction, reference)

            self.optimizer.zero_grad()
            loss.backward()
            losses.append(loss.item())

            if (i % self.train_batchsize) == 0:
                self.optimizer.step()
            pbar.set_postfix({'loss': losses[-1]})
        avg_loss = float(np.mean(losses).item())
        self.epoch += 1
        return avg_loss

    def evaluate(self, cluster_idxs: np.ndarray, reference_file: os.path, read_dir: os.path):
        os.makedirs(self.prediction_dir, exist_ok=True)
        self.net.eval()
        dna_bases = DNA_BASES + ['N']

        processed_idxs = [int(filename[:-4]) for filename in os.listdir(self.prediction_dir)]
        unprocessed_idxs = np.setdiff1d(cluster_idxs, processed_idxs).reshape(-1)
        if len(unprocessed_idxs) > 0:
            dataset = Dataset(cluster_idxs=unprocessed_idxs, reference_file=reference_file, read_dir=read_dir,
                              window_size=self.config['net']['window']['window size'])
            dataloader = tdata.DataLoader(dataset=dataset, batch_size=1, collate_fn=dataset.collate_fn,
                                          num_workers=NUM_WORKERS, shuffle=False, drop_last=False)

            print('INFO: predicting DNA strands...')
            pbar = tqdm(dataloader)
            with torch.no_grad():
                for i, (forward_reads, backward_reads, reference) in enumerate(pbar):
                    prediction_length = reference.shape[0]

                    prediction = self.get_prediction(forward_reads=forward_reads, backward_reads=backward_reads,
                                                     prediction_length=prediction_length)
                    prediction = torch.argmax(prediction, dim=1).reshape(-1).cpu().numpy()
                    prediction = [dna_bases[idx] for idx in prediction]

                    prediction_file = os.path.join(self.prediction_dir, '{}.npy'.format(unprocessed_idxs[i]))
                    np.save(prediction_file, prediction)

        print('INFO: evaluating...')
        nbases, nstrands, base_acc, strand_acc = 0, 0, 0., 0.
        with open(reference_file, 'r') as f:
            references = f.readlines()
        references = [np.array(list(reference.strip())) for reference in references]

        for cluster_id in cluster_idxs:
            prediction_file = os.path.join(self.prediction_dir, '{}.npy'.format(cluster_id))
            prediction = np.load(prediction_file)
            reference = references[cluster_id]

            prediction = [dna_bases.index(base) for base in prediction]
            reference = [dna_bases.index(base) for base in reference]

            assert len(prediction) == len(reference), '{} {}'.format(len(prediction), len(reference))
            is_equal = np.equal(prediction, reference)

            base_acc += np.sum(is_equal).item()
            nbases += len(is_equal)

            strand_acc += np.all(is_equal).item()
            nstrands += 1
        base_acc /= nbases
        strand_acc /= nstrands

        results = {
            'base accuracy': float(base_acc),
            'strand accuracy': float(strand_acc)
        }
        result_file = os.path.join(self.result_dir, 'results.json')
        with open(result_file, 'w') as f:
            json.dump(results, f)

        message = 'INFO:'
        for k, v in results.items():
            message += ' {} {};'.format(k, v)
        print(message)
        notifyhub.send(message=message, config_fp=NOTIFYHUB_FP)
        return results

    def get_prediction(self, forward_reads: torch.Tensor, backward_reads: torch.Tensor,
                       prediction_length: int) -> torch.Tensor:
        """

        Args:
            forward_reads: R x T x 4 reads
            backward_reads: R x T x 4 reads
            prediction_length: integer indicating length of output prediction

        Returns:

        """
        num_reads, _, _ = forward_reads.shape
        assert forward_reads.shape == backward_reads.shape

        reads = torch.stack([forward_reads, backward_reads], dim=0)
        reads = reads.to(self.device)
        out = self.net(reads, prediction_length)

        backward_length = prediction_length // 2
        forward_length = prediction_length - backward_length

        forward_prediction = out[0, :forward_length]
        backward_prediction = torch.flip(out[1, :backward_length], dims=[0])
        prediction = torch.cat([forward_prediction, backward_prediction], dim=0)
        assert prediction.shape[0] == prediction_length
        return prediction


@notifyhub.watch(config_fp=NOTIFYHUB_FP)
def main():
    argparser = argparse.ArgumentParser()
    argparser.add_argument('--device', type=int, choices=list(range(torch.cuda.device_count())), default=0)
    argparser.add_argument('--config', type=str, required=True)
    args = argparser.parse_args()

    device = torch.device(args.device)
    config = args.config
    trainer = Trainer(config=config, device=device)
    trainer.train()


if __name__ == '__main__':
    main()
