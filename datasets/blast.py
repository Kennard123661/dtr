import os
import numpy as np
from tqdm import tqdm

from directory import BLAST_DIR as DATASET_DIR

RAW_DATASET_FILE = os.path.join(DATASET_DIR, 'combined_djordje_with_references.txt')
PROCESSED_DIR = os.path.join(DATASET_DIR, 'processed')
REFERENCE_FILE = os.path.join(PROCESSED_DIR, 'references.txt')
READ_DIR = os.path.join(PROCESSED_DIR, 'noisy-reads')
SPLIT_DIR = os.path.join(PROCESSED_DIR, 'splits')
NSPLITS = 5


def read_raw_file() -> (list, list):
    print('INFO: reading raw dataset file...')
    with open(RAW_DATASET_FILE, 'r') as f:
        lines = f.readlines()
    lines = [line.strip() for line in lines]

    is_nstrand = np.zeros(len(lines), dtype=bool)
    for i, expected_nstrands in enumerate(lines):
        try:
            is_nstrand[i] = isinstance(int(expected_nstrands), int)
        except:
            pass

    # find the idxs that are integers
    idxs = np.sort(np.argwhere(is_nstrand)).reshape(-1)
    nreferences = idxs.shape[0]
    idxs = np.concatenate([idxs, np.ones(1, dtype=int) * len(lines)])

    all_references, all_reads = [], []
    for i in tqdm(range(nreferences)):
        start, end = idxs[i], idxs[i+1]
        reference = str(lines[start+1])
        reads = [str(line) for line in lines[start+2:end]]
        assert np.all([str(read).isalpha() for read in reads]) and reference.isalpha()

        all_references.append(reference)
        all_reads.append(reads)
    assert len(all_references) == len(all_reads)
    return all_references, all_reads


def process_dataset():
    print('INFO: creating references and reads...')
    all_references, all_reads = read_raw_file()
    os.makedirs(READ_DIR, exist_ok=True)

    with open(REFERENCE_FILE, 'w') as f:
        for reference in all_references:
            f.write(reference + '\n')

    for i, reads in enumerate(all_reads):
        save_file = os.path.join(READ_DIR, '{}.txt'.format(i))
        with open(save_file, 'w') as f:
            for read in reads:
                f.write(read + '\n')

    print('INFO: creating evaluation splits...')
    os.makedirs(SPLIT_DIR, exist_ok=True)
    nreferences = len(all_references)
    idxs = np.arange(nreferences)
    np.random.shuffle(idxs)
    for i in range(NSPLITS):
        save_file = os.path.join(SPLIT_DIR, 'split-{}.npy'.format(i))
        test_idxs = np.sort(idxs[i::NSPLITS])
        np.save(save_file, test_idxs)


def main():
    process_dataset()


if __name__ == '__main__':
    main()
