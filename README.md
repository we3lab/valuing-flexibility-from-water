# Valuing Energy Flexibility from Water Systems

<img src="https://github.com/we3lab/valuing-flexibility-from-water/blob/main/figures/png/figure2.png" width="50%" align="right">

This repository contains data and plotting functions associated with the paper titled: *Valuing Energy Flexibility from Water Systems*.


The associated manuscript is accepted and awaiting publication in Nature Water.


An web-app with interactive plots is available at: https://lvof.we3lab.tech

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
marimo run marimonotebook/lvof.py
```
If using a conda environment, ensure the environment is activated before running marimo.

## Data
Configuration options for plotting include:

Case Name : Tariff structure used for flexible operating schema optimization and technoeconomic calculations.

System Type : Advanced Water Treatment (desalination), Water Distribution, or Wastewater treatment.

Plot Type: ```timeseries```, ```radar```, or ```costing```.

Representative day: Single day snapshot (or annualized versions) associated with each tariff structure.
