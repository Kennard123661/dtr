import sys
import os


if __name__ == '__main__':
    PROJECT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    sys.path.insert(0, PROJECT_DIR)


def process_dataset(dataset: str):
    if dataset == 'blast':
        from datasets import blast as dset
    else:
        raise ValueError('no such dataset {}'.format(dataset))
    dset.process_dataset()


def main():
    for dataset in ['blast']:
        process_dataset(dataset=dataset)


if __name__ == '__main__':
    main()
