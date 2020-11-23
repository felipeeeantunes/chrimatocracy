# %%
import pandas as pd
import glob
import os
import unidecode
from pathlib import Path
import sys
import os
parent_dir = Path().cwd()
# %%
raw_data_path = Path(parent_dir) / 'raw_data/2014/'
data_path = Path(parent_dir) / 'data/'

if not os.path.exists(data_path):
    os.makedirs(data_path)


print('Getting info about dirty CPF/CNPJ...')


# %% [markdown]
# ### Comparing with CNEP and CEIS data.
# - http://www.portaltransparencia.gov.br/download-de-dados/cnep
# - http://www.portaltransparencia.gov.br/download-de-dados/ceis

# %%
CNEP = pd.read_csv('http://www.portaltransparencia.gov.br/download-de-dados/cnep/20201029',
                   compression='zip', sep=';', encoding='iso-8859-1')
CEIS = pd.read_csv('http://www.portaltransparencia.gov.br/download-de-dados/ceis/20201029',
                   compression='zip', sep=';', encoding='iso-8859-1')
CNEP.to_csv(data_path / '20201029_CNEP.csv', index=False)
CEIS.to_csv(data_path / '20201029_CEIS.csv', index=False)

print('Done.')

