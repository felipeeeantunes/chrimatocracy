# Chrimatocracy
Data Science applied to the government sector

## Build
You'll need to install [Poetry](https://python-poetry.org/) and set a list of environment variables:

### Poetry Installation
We are using the `1.1.10` version, type:
```bash
curl -sSL https://install.python-poetry.org | python3 - --version 1.1.10
```
Add `export PATH="/Users/<user.name>/.local/bin:$PATH"` to your shell configuration file (eg. ~/.zshrc) and restart your terminal and check if Poetry was installed correctly.

```bash
poetry --version
```

## Straightforward instructions for setting up and running Chirmatocracy

These are provisory instructions for working with provisory branches.


1. Clone the project:
    ```bash
    git clone https://github.com/felipeeeantunes/chrimatocracy.git
    ```

2. Switch to the specific branch you are going to work on (`feature/structuring-modules`, for instance):
    ```bash
    git switch feature/structuring-modules
    ```

3. Go inside the directory and build the environment:
    ```bash
    cd chrimatocracy
    make build
    ```
    Ps. You can use the Poetry commands to create local environments and builds instead.

4. Code patterns
   ```bash
   # run these commands below before adding files to commit
   make isort
   make black
   # then, git add <files>...
   ```

## Using the library for Reseach
Under the research folder you find two applications of the library:

1. `donations` contains the code of the paper [*Statistical analysis of Brazilian electoral campaigns via Benford’s law*](https://www.sciencedirect.com/science/article/abs/pii/S0378437117313699) published at Physica A: Statistical Mechanics and its Applications. The preprint can be found on Arxiv as [*Evidence of Fraud in Brazil's Electoral Campaigns Via the Benford's Law*](https://arxiv.org/abs/1707.08826)

2. `network` contains the code behind an ongoing research project called *Detecting Anomaly donations by Combining Benford's Law and Social Network Analysis*.

Also, you can find, under `notebooks` Jupyter notebooks (outdated) with descriptive analyses of the data.


## Data
You can find the data in raw format and already prepared (using `data_preparation` scripts) [here](https://www.dropbox.com/sh/gkoazh77kcp4nh7/AAC-HgkF0R08SU3bNRCK7ho8a?dl=0). The raw data is public and was obtained from the [TSE website](https://divulgacandcontas.tse.jus.br/divulga/#/2014).

Data from CEIS and CNEP (containing companies with restrictions) was obtained from Portal da Transparência and the links are:
- CEIS: http://www.portaltransparencia.gov.br/download-de-dados/cnep/20201029
- CNEP: http://www.portaltransparencia.gov.br/download-de-dados/ceis/20201029