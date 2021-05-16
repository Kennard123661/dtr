import os
import platform


hostname = platform.node()
if hostname == 'kennardng-deskop':
    PROJECT_DIR = '/home/kennardng/projects/dtr'
    SAVE_DIR = '/mnt/Data/project-storage/dtr'
else:
    raise ValueError('no such hostname {}'.format(hostname))

DATASET_DIR = os.path.join(SAVE_DIR, 'datasets')
BLAST_DIR = os.path.join(DATASET_DIR , 'blast')
