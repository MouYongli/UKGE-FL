
# Enhancing Uncertain Knowledge Graphs Embedding using Fuzzy Logic

## Dataset Setup

We use three uncertain knowledge graph:
- CN15k
- NL27k
- PPI5k

downdgoqng

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