# MetalgPy + gnnNAS

## About
The project focusses on leveraging the general purpose library [MetalgPy](https://github.com/deephyper/metalgpy) to write symbolized ML programs capable of leveraging graph hyperparameters for better surrogate model fitting. Our goal was to use `MetalgPy` to search for a representation learning algorithm for graph structures.

## Packages

- `PyTorch`
- `PyTorch-Geometric`
- `MetalgPy`

```shell
# Install Pytorch
pip install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu116

# Install Pytorch Geometric
pip install -q torch-scatter -f https://data.pyg.org/whl/torch-${TORCH}.html
pip install -q torch-sparse -f https://data.pyg.org/whl/torch-${TORCH}.html
pip install -q git+https://github.com/pyg-team/pytorch_geometric.git

# Install DeepHyper/MetalgPy
pip install -q metalgpy

# Install rdkit for the datasets
pip install -q rdkit-pypi
```

## Datasets

We use three benchmark datasets

- `GNN Benchmark Dataset` 
- `Planetoid-1`
- `MoleculeNet`

#### GNN Benchmark Dataset

A variety of artificially and semi-artificially generated graph datasets. It is composed of datasets such as `PATTERN`, `CLUSTER`, `MNIST`, `CIFAR-10`, `TSP`, `CSL`. 

`Reference`: https://arxiv.org/abs/2003.00982
`Resource`: https://pytorch-geometric.readthedocs.io/en/latest/modules/datasets.html

#### Planetoid-1:

`Planetoid` dataset comprising of citation network datasets `Cora`, `Citeseer`, and `Pubmed`. These are three benchmark datasets used for semi-supervised node classification tasks. Each of the mentioned graph datasets contains bag-of-words representation of documents and citation links between the documents

`Reference`: https://arxiv.org/pdf/1603.08861.pdf
`Resource`: https://pytorch-geometric.readthedocs.io/en/latest/modules/datasets.html

#### MoleculeNet:

`MoleculetNet`: MoleculeNet is a benchmark specially designed for testing machine learning methods of molecular properties. As we aim to facilitate the development of molecular machine learning method, this work curates a number of dataset collections, creates a suite of software that implements many known featurizations and previously proposed algorithms. All methods and datasets are integrated as parts of the open source DeepChem package(MIT license).

Within the `MoleculetNet`, we are interested to benchmark, Quantum Mechanics, and Physical chemistry datasets

`Quantum Mechanics`:
- QM7/QM7b (structure): Electronic properties(atomization energy, HOMO/LUMO, etc.) determined using ab-initio density functional theory(DFT).
- QM8 (structure): Electronic spectra and excited state energy of small molecules calculated by multiple quantum mechanic methods.
- QM9 (structure): Geometric,  energetic, electronic and thermodynamic properties of DFT-modelled small molecules.

`Physical chemistry`:
- ESOL: Water solubility data(log solubility in mols per litre) for common organic small molecules.
- FreeSolv: Experimental and calculated hydration free energy of small molecules in water.
- Lipophilicity: Experimental results of octanol/water distribution coefficient(logD at pH 7.4).

`Reference`: https://moleculenet.org/datasets-1
`Resource`: https://pytorch-geometric.readthedocs.io/en/latest/modules/datasets.html

## Results
TBA

## Author
[Akhil Pandey](https://github.com/akhilpandey95)

## Supervisor
[Prasanna Balaprakash](https://github.com/pbalapra)

