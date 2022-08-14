#%% [markdown]
# # Investigação das doações para Deputado Estadual no Brasil
#

import gc
import os
#%%
import sys
from pathlib import Path

parent_dir = Path().cwd()
print((Path(parent_dir) / "assets/"))
sys.path.append(str(Path(parent_dir) / "assets/"))


import pandas as pd
import seaborn as sns

pd.options.display.float_format = "{:.2f}".format

#%%
raw_data_path = Path(parent_dir) / "raw_data/2014/"
data_path = Path(parent_dir) / "data/"
table_path = Path(parent_dir) / "tables/"
figure_path = Path(parent_dir) / "figures/"

directories = [table_path, figure_path]

for directory in directories:
    if not os.path.exists(directory):
        os.makedirs(directory)


import matplotlib as mpl
#%%
import matplotlib.pyplot as plt

mpl.use("pgf")
pgf_with_pdflatex = {
    "pgf.texsystem": "pdflatex",
    "pgf.preamble": "\n".join(
        [
            r"\usepackage[utf8x]{inputenc}",
            r"\usepackage[T1]{fontenc}",
            r"\usepackage{cmbright}",
        ]
    ),
}
mpl.rcParams.update(pgf_with_pdflatex)
# get_ipython().run_line_magic('matplotlib', 'inline')

rc = {
    "savefig.dpi": 75,
    "figure.autolayout": False,
    "figure.figsize": [12, 8],
    "axes.labelsize": 18,
    "axes.titlesize": 18,
    "font.size": 18,
    "lines.linewidth": 2.0,
    "lines.markersize": 8,
    "legend.fontsize": 16,
    "xtick.labelsize": 16,
    "ytick.labelsize": 16,
}

sns.set(style="whitegrid", rc=rc)

data_types = {
    "id_accountant_cnpj": str,
    "id_candidate_cpf": str,
    "id_donator_cpf_cnpj": str,
    "id_original_donator_cpf_cnpj": str,
    "id_donator_effective_cpf_cnpj": str,
}

comites_brasil = pd.read_csv(data_path / "brazil_2014_donations_committees.csv", dtype=data_types)
partidos_brasil = pd.read_csv(data_path / "brazil_2014_donations_parties.csv", dtype=data_types)
corruptos_brasil = pd.read_csv(data_path / "brazil_2014_donations_candidates.csv", dtype=data_types)

#%% [markdown]
# #### Partidos

#%%
print("Número total de doações a partidos:", partidos_brasil["cat_party"].count())


#%%
print("Número de partidos que receberam doações:", partidos_brasil["cat_party"].nunique())


#%%
diretas_part_pj = partidos_brasil[
    (partidos_brasil["id_original_donator_cpf_cnpj"].isnull())
    & (partidos_brasil["id_donator_cpf_cnpj"].astype(str).apply(lambda x: len(x)) == 14)
]

print("Número de doações diretas de empresas a partidos:", diretas_part_pj["cat_party"].count())
print("Valor: R$", "%.1e" % diretas_part_pj["num_donation_ammount"].sum())


#%%
diretas_part_pf = partidos_brasil[
    (partidos_brasil["id_original_donator_cpf_cnpj"].isnull())
    & (partidos_brasil["id_donator_cpf_cnpj"].astype(str).apply(lambda x: len(x)) == 11)
]

print("Número de doações diretas de indivíduos a partidos:", diretas_part_pf["cat_party"].count())
print("Valor: R$", "%.1e" % diretas_part_pf["num_donation_ammount"].sum())


#%%
indiretas_part_pj = partidos_brasil[
    (partidos_brasil["id_original_donator_cpf_cnpj"].astype(str).apply(lambda x: len(x)) == 14)
    & (partidos_brasil["id_donator_cpf_cnpj"].astype(str).apply(lambda x: len(x)) == 14)
]


print("Número de doações indiretas provenientes de empresas a partidos:", indiretas_part_pj["cat_party"].count())
print("Valor: R$", "%.1e" % indiretas_part_pj["num_donation_ammount"].sum())


#%%
indiretas_part_pf = partidos_brasil[
    (partidos_brasil["id_original_donator_cpf_cnpj"].astype(str).apply(lambda x: len(x)) == 11)
    & (partidos_brasil["id_donator_cpf_cnpj"].astype(str).apply(lambda x: len(x)) == 14)
]

print("Número de doações indiretas provenientes de individuos a partidos:", indiretas_part_pf["cat_party"].count())
print("Valor: R$", "%.1e" % indiretas_part_pf["num_donation_ammount"].sum())


#%%
unknow_part = partidos_brasil[
    (partidos_brasil["cat_original_donator_name"].isnull()) & (partidos_brasil["id_donator_cpf_cnpj"].isnull())
]
# corruptos_brasil.drop(unknow.index, inplace=True)

print("Número de doações não rastreadas a partidos", unknow_part["cat_party"].count())
print("Valor: R$", "%.1e" % unknow_part["num_donation_ammount"].sum())

#%% [markdown]
# #### Comitês

#%%
# print(comites_brasil['cat_commitee_type'].unique())


#%%
print("Número total de doações a comites:", comites_brasil["id_commitee_seq"].count())


#%%
print("Número de comites que receberam doações:", comites_brasil["id_commitee_seq"].nunique())


#%%
diretas_com_pj = comites_brasil[
    (comites_brasil["id_original_donator_cpf_cnpj"].isnull())
    & (comites_brasil["id_donator_cpf_cnpj"].astype(str).apply(lambda x: len(x)) == 14)
]

print("Número de doações diretas de empresas a comites:", diretas_com_pj["id_commitee_seq"].count())
print("Valor: R$", "%.1e" % diretas_com_pj["num_donation_ammount"].sum())


#%%
diretas_com_pf = comites_brasil[
    (comites_brasil["id_original_donator_cpf_cnpj"].isnull())
    & (comites_brasil["id_donator_cpf_cnpj"].astype(str).apply(lambda x: len(x)) == 11)
]

print("Número de doações diretas de indivíduos a comites:", diretas_com_pf["id_commitee_seq"].count())
print("Valor: R$", "%.1e" % diretas_com_pf["num_donation_ammount"].sum())


#%%
indiretas_com_pj = comites_brasil[
    (comites_brasil["id_original_donator_cpf_cnpj"].astype(str).apply(lambda x: len(x)) == 14)
    & (comites_brasil["id_donator_cpf_cnpj"].astype(str).apply(lambda x: len(x)) == 14)
]


print("Número de doações indiretas provenientes de empresas a comites:", indiretas_com_pj["id_commitee_seq"].count())
print("Valor: R$", "%.1e" % indiretas_com_pj["num_donation_ammount"].sum())


#%%
indiretas_com_pf = comites_brasil[
    (comites_brasil["id_original_donator_cpf_cnpj"].astype(str).apply(lambda x: len(x)) == 11)
    & (comites_brasil["id_donator_cpf_cnpj"].astype(str).apply(lambda x: len(x)) == 14)
]

print("Número de doações indiretas provenientes de individuos a comites:", indiretas_com_pf["id_commitee_seq"].count())
print("Valor: R$", "%.1e" % indiretas_com_pf["num_donation_ammount"].sum())


#%%
unknow_com = comites_brasil[
    (comites_brasil["cat_original_donator_name"].isnull()) & (comites_brasil["id_donator_cpf_cnpj"].isnull())
]
# corruptos_brasil.drop(unknow.index, inplace=True)

print("Número de doações não rastreadas a comites", unknow_com["id_commitee_seq"].count())
print("Valor: R$", "%.1e" % unknow_com["num_donation_ammount"].sum())


#%%
print("Número total de doações a candidatos:", corruptos_brasil["id_candidate_cpf"].count())


#%%
print("Número de candidatos que receberam doações:", corruptos_brasil["id_candidate_cpf"].nunique())


#%%
diretas_pj = corruptos_brasil[
    (corruptos_brasil["id_original_donator_cpf_cnpj"].isnull())
    & (corruptos_brasil["id_donator_cpf_cnpj"].astype(str).apply(lambda x: len(x)) == 14)
]

print("Número de doações diretas de empresas:", diretas_pj["id_candidate_cpf"].count())
print("Valor: R$", "%.1e" % diretas_pj["num_donation_ammount"].sum())


#%%
diretas_pf = corruptos_brasil[
    (corruptos_brasil["id_original_donator_cpf_cnpj"].isnull())
    & (corruptos_brasil["id_donator_cpf_cnpj"].astype(str).apply(lambda x: len(x)) == 11)
]

print("Número de doações diretas de indivíduos:", diretas_pf["id_candidate_cpf"].count())
print("Valor: R$", "%.1e" % diretas_pf["num_donation_ammount"].sum())


#%%
indiretas_pj = corruptos_brasil[
    (corruptos_brasil["id_original_donator_cpf_cnpj"].astype(str).apply(lambda x: len(x)) == 14)
    & (corruptos_brasil["id_donator_cpf_cnpj"].astype(str).apply(lambda x: len(x)) == 14)
]

print("Número de doações indiretas provenientes de empresas:", indiretas_pj["id_candidate_cpf"].count())
print("Valor: R$", "%.1e" % indiretas_pj["num_donation_ammount"].sum())


#%%
indiretas_pf = corruptos_brasil[
    (corruptos_brasil["id_original_donator_cpf_cnpj"].astype(str).apply(lambda x: len(x)) == 11)
    & (corruptos_brasil["id_donator_cpf_cnpj"].astype(str).apply(lambda x: len(x)) == 14)
]

print("Número de doações indiretas provenientes de individuos:", indiretas_pf["id_candidate_cpf"].count())
print("Valor: R$", "%.1e" % indiretas_pf["num_donation_ammount"].sum())


#%%
indiretas_teste = corruptos_brasil[
    (corruptos_brasil["id_original_donator_cpf_cnpj"].isnull() == False)
    & (corruptos_brasil["id_donator_cpf_cnpj"].astype(str).apply(lambda x: len(x)) == 11)
]


print("Número de doações com intermediário sendo uma pessoa física", indiretas_teste["id_candidate_cpf"].count())
print("Valor: R$", "%.1e" % indiretas_teste["num_donation_ammount"].sum())
if indiretas_teste["id_candidate_cpf"].count() == 0:
    print("UFA!")


#%%
unknow = corruptos_brasil[
    (corruptos_brasil["cat_original_donator_name"].isnull()) & (corruptos_brasil["id_donator_cpf_cnpj"].isnull())
]
# corruptos_brasil.drop(unknow.index, inplace=True)

print("Número de doações não rastreadas", unknow["id_candidate_cpf"].count())
print("Valor: R$", "%.1e" % unknow["num_donation_ammount"].sum())
