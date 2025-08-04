# Valuing Energy Flexibility from Water Systems

<img src="https://github.com/we3lab/valuing-flexibility-from-water/blob/main/figures/png/figure2.png" width="50%" align="right">

This repository contains data and plotting functions associated with the paper titled: *Valuing Energy Flexibility from Water Systems*.

The associated manuscript is published in [Nature Water](https://www.nature.com/articles/s44221-024-00316-4).

Interactive plots can be accessed at https://lvof.we3lab.tech.

## Interacting with data
**1. Download the source code**

In your local machine, navigate to a desired directory and run:
```
git clone https://github.com/we3lab/valuing-flexibility-from-water.git
```
Alternatively, you may download the source code from github and unzip in the desired location.

**2. Install dependencies**

Conda:
```
conda env create -f environment.yml
```
Pip:
```
pip install -r requirements.txt
```

**3. Run notebook locally**

```
cd marimonotebook
marimo run lvof.py
```
If using a conda environment, ensure the environment is activated before running marimo.

## Data
Configuration options for plotting include:

Case Name : Tariff structure used for flexible operating schema optimization and technoeconomic calculations.

System Type : Advanced Water Treatment (desalination), Water Distribution, or Wastewater treatment.

Plot Type: ```timeseries```, ```radar```, or ```costing```.

Representative day: Single day snapshot (or annualized versions) associated with each tariff structure.

# Attribution & Acknowledgements

As mentioned above, the manuscript associated with this repository can be found in [Nature Water](https://www.nature.com/articles/s44221-024-00316-4). The citation is available in [CITATION.cff](https://github.com/we3lab/valuing-flexibility-from-water/blob/main/CITATION.cff) or in BibTeX format below:

```
@article{rao2024valuing,
  title={Valuing energy flexibility from water systems},
  author={Rao, Akshay K and Bolorinos, Jose and Musabandesu, Erin and Chapin, Fletcher T and Mauter, Meagan S},
  journal={Nature Water},
  volume={2},
  number={10},
  pages={1028--1037},
  year={2024},
  month={Sep},
  day={27},
  publisher={Nature Publishing Group UK London}
}
```

This work was supported by the following grants and programs:

- [Department of Energy, the Office of Energy Efficiency and Renewable Energy, Advanced Manufacturing Office](https://www.energy.gov/eere/ammto/advanced-materials-and-manufacturing-technologies-office) (grant number DE-EE0009499)
- [National Alliance for Water Innovation (NAWI)](https://www.nawihub.org/) (grant number UBJQH - MSM)

The views expressed herein do not necessarily represent the views of the US Department of Energy or the United States Government. 

We thank A. Atia and T. Bartholomew from the National Energy Technology Laboratory; B. Knueven from the National Renewable Energy Laboratory; A. Miot and A. Akela from Silicon Valley Clean Water; J. Haggmark, G. Paul and B. Rahrer from the City of Santa Barbara; A. Dudchenko from SLAC National Accelerator Laboratory and S. A. Farraj for their helpful conversations and feedback on the work.

