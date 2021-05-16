import os
import platform


hostname = platform.node()
if hostname == 'kennardng-desktop':
    PROJECT_DIR = '/home/kennardng/projects/dtr'
    SAVE_DIR = '/mnt/Data/project-storage/dtr'
else:
    raise ValueError('no such hostname {}'.format(hostname))

CONFIG_DIR = os.path.join(PROJECT_DIR, 'configs')
NOTIFYHUB_FP = os.path.join(CONFIG_DIR, 'notifyhub.json')
DATASET_DIR = os.path.join(SAVE_DIR, 'datasets')
BLAST_DIR = os.path.join(DATASET_DIR , 'blast')
