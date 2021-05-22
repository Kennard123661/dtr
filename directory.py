import os
import platform


hostname = platform.node()
if hostname == 'kennardng-desktop':
    PROJECT_DIR = '/home/kennardng/projects/dtr'
    DATA_DIR = '/mnt/Data/project-storage/dtr'
elif hostname == 'xgpe2':
    PROJECT_DIR = '/home/e/e0036319/projects/dtr'
    DATA_DIR = '/home/e/e0036319/project-storage/dtr'
else:
    raise ValueError('no such hostname {}'.format(hostname))

CONFIG_DIR = os.path.join(PROJECT_DIR, 'configs')
SAVE_DIR = os.path.join(DATA_DIR, 'saved')

NOTIFYHUB_FP = os.path.join(CONFIG_DIR, 'notifyhub.json')
DATASET_DIR = os.path.join(DATA_DIR, 'datasets')
BLAST_DIR = os.path.join(DATASET_DIR, 'blast')
