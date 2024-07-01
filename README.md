
# Enhancing Uncertain Knowledge Graphs Embedding using Fuzzy Logic

## Dataset Setup

We use the following 3 uncertain knowledge graph datasets:
- CN15k
- NL27k
- PPI5k

Datasets are from other GitHub repos from [ShihanYang/UKGsE](https://github.com/ShihanYang/UKGsE.git) (train/test) and [stasl0217/UKGE](https://github.com/stasl0217/UKGE/tree/master) (train/val/test).
We need to preprocess NL27K dataset, Code available [here](./scripts/preprocess_nl27k.py)

## Python Environment Setup

1. conda environment
```
conda create --name ukge python=3.11
conda activate ukge
```

2. jupyter lab and kernel
```
conda install -c conda-forge jupyterlab
conda install ipykernel
ipython kernel install --user --name=ukge
jupyter lab --no-browser --port=8888
```

```
pip install torch==2.2.1 torchvision==0.17.1 torchaudio==2.2.1
pip install torch_geometric
pip install pyg_lib torch_scatter torch_sparse torch_cluster torch_spline_conv -f https://data.pyg.org/whl/torch-2.2.0+cu121.html
pip install -r requirements.txt
pip install -e .
```