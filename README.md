

## Setup

### Environment

```bash
conda create -n dtr python==3.7.6
conda activate dtr
chmod +x setup.sh
./setup.sh
```


## Scripts

### Processing datasets

```bash
python scripts/process_datasets.py
```


### Training


Run the following script with the `dtr` activated.

```bash
python scripst/train_dtr_double.py --config base
```
