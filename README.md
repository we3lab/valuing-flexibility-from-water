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

uv (for [marimo notebooks](https://marimo.io/)) - *Recommended*

Installing [uv](https://github.com/astral-sh/uv) allows you to run the notebooks which have inline scripts containing the required packages in an [isolated virtual environment.](https://marimo.io/blog/sandboxed-notebooks). The `marimonotebook\lvof.py` notebook in this repo has it's package dependencies inlined.

```
pip install uv
```

Conda:

Ensure the conda environment is activated before running marimo.

```
conda env create -f environment.yml
```
Pip:
```
pip install -r requirements.txt
```

**3. Run notebook locally**

uv installed:

```
cd marimonotebook
uvx marimo run --sandbox lvof.py
```

To edit the notebook source code, replace `run` with `edit` in the above commands.

If you don't have `uv` installed, you can use:

```shell
cd marimonotebook
marimo run lvof.py
```

> [!NOTE]
> You may need to manually install dependencies if not using `uv`.

## Data
Configuration options for plotting include:

Case Name : Tariff structure used for flexible operating schema optimization and technoeconomic calculations.

System Type : Advanced Water Treatment (desalination), Water Distribution, or Wastewater treatment.

Plot Type: ```timeseries```, ```radar```, or ```costing```.

Representative day: Single day snapshot (or annualized versions) associated with each tariff structure.
