# %%
import glob
import os
import sys
from pathlib import Path

import pandas as pd
import unidecode

cwd = Path().absolute()

year = 2014
data_path = Path(cwd) / "data" / "prepared" / f"{year}/"

if not os.path.exists(data_path):
    os.makedirs(data_path)


print("Getting info about dirty CPF/CNPJ...")


# %% [markdown]
# ### Comparing with CNEP and CEIS data.
# - http://www.portaltransparencia.gov.br/download-de-dados/cnep
# - http://www.portaltransparencia.gov.br/download-de-dados/ceis

# %%
CNEP = pd.read_csv(
    "https://www.portaltransparencia.gov.br/download-de-dados/cnep/20220826",
    compression="zip",
    sep=";",
    encoding="iso-8859-1",
)
CEIS = pd.read_csv(
    "http://www.portaltransparencia.gov.br/download-de-dados/ceis/20220826",
    compression="zip",
    sep=";",
    encoding="iso-8859-1",
)
CNEP.to_csv(data_path / "20220826_CNEP.csv", index=False)
CEIS.to_csv(data_path / "20220826_CEIS.csv", index=False)

print("Done.")
