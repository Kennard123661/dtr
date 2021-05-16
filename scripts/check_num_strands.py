import argparse
import numpy as np


def check_nstrands(file: str) -> (list, list):
    print('INFO: checking raw dataset file {}...'.format(file))
    with open(file, 'r') as f:
        lines = f.readlines()
    lines = [line.strip() for line in lines]

    is_nstrand = np.zeros(len(lines), dtype=bool)
    for i, line in enumerate(lines):
        try:
            is_nstrand[i] = isinstance(int(line), int)
        except:
            pass

    # find the idxs that are integers
    idxs = np.sort(np.argwhere(is_nstrand)).reshape(-1)
    nreferences = idxs.shape[0]
    idxs = np.concatenate([idxs, np.ones(1, dtype=int) * len(lines)])

    for i in range(nreferences):
        start, end = idxs[i], idxs[i+1]
        reference = str(lines[start+1])
        reads = [str(line) for line in lines[start+2:end]]

        expected_nstrands = int(lines[start])
        actual_nstrands = len(reads)

        if actual_nstrands != expected_nstrands:
            print('cluster idx: {}; actual: {}; expected: {}'.format(i, actual_nstrands, expected_nstrands))
        assert np.all([str(read).isalpha() for read in reads]) and reference.isalpha()


def main():
    argparser = argparse.ArgumentParser()
    argparser.add_argument('--file', type=str, required=True, help='path to the dna read file.')
    args = argparser.parse_args()
    check_nstrands(file=args.file)


if __name__ == '__main__':
    main()
