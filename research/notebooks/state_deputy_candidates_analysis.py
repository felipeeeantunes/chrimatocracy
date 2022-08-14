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

from matplotlib.backends.backend_pgf import FigureCanvasPgf
mpl.backend_bases.register_backend('pdf', FigureCanvasPgf)

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


#%% [markdown]
# ##  Informações sobre a votação e o perfil dos candidatos
#%% [markdown]
# A tabela de candidatos possui informações sobre o perfil dos candidatos,tais como raça, escolaridade, estado civil,
# além das informações relacionadas às eleições.  Além disso, para associarmos a lista de votação à lista de doações, precisamos da informação sobre o CPF, que só consta na tabela de candidatos. Esse mapeamento somente pode ser feito através da sequencial, constante tanto na tabela de candidatos quando na de votos.

# candidatos_brasil = pd.read_csv(data_path / 'brazil_2014_candidates.csv')exit
#%% [markdown]
# Filtramos candidatos para os cargos estudos, cuja candidatura foi deferida

#%%
candidatos_deputado_estadual_indeferidos = pd.read_csv(
    data_path / "brazil_2014_state_deputy_candidates_non_accepted.csv", dtype=data_types
)
print("Cargos:", candidatos_deputado_estadual_indeferidos["cat_role_description"].unique())
print("Número de canidatos indeferidos", candidatos_deputado_estadual_indeferidos["id_candidate_sequential"].nunique())


#%%
candidatos_deputado_estadual_deferidos = pd.read_csv(
    data_path / "brazil_2014_state_deputy_candidates_accepted.csv", dtype=data_types
)
print("Cargos:", candidatos_deputado_estadual_deferidos["cat_role_description"].unique())
print("Número de candidatos deferidos", candidatos_deputado_estadual_deferidos["id_candidate_sequential"].nunique())


#%% [markdown]
# Criamos a lista de candidatos baseado na sua sequencial, utilizada para cruzar essa base de informações com a base votação.

#%%
list_cand = candidatos_deputado_estadual_deferidos["id_candidate_sequential"].unique()
print("Número de candidatos na lista:", len(list_cand))

#%% [markdown]
# A tabela de votos possui informação sobre os dados apurados da votação, no primeiro e segundo turno, em cada munício, além de informações eleitorias.
# votos_brasil = pd.read_csv(data_path / 'brazil_2014_votes.csv')

#%%
votos_deputados_estaduais_indeferidos = pd.read_csv(
    data_path / "brazil_2014_state_deputy_candidates_non_accepted_votes.csv", dtype=data_types
)
print("Cargos:", votos_deputados_estaduais_indeferidos["cat_role_description"].unique())
print("Número de candidatos indeferidos", votos_deputados_estaduais_indeferidos["id_candidate_sequential"].nunique())


#%%
votos_deputados_estaduais_deferidos = pd.read_csv(
    data_path / "brazil_2014_state_deputy_candidates_accepted_votes.csv", dtype=data_types
)
print("Cargos:", votos_deputados_estaduais_deferidos["cat_role_description"].unique())
print("Número de candidatos deferidos", votos_deputados_estaduais_deferidos["id_candidate_sequential"].nunique())


#%% [markdown]
# Percebemos que há um candidato a mais do que o da lista de candidatos. É um fato curioso um candidato receber votos, mas não constar na lista inicial.
# Trata-se de Tâmara Joana Biolo Soares, que não foi eleita.

#%% [markdown]
# Vamos user uma tabela contendo informações do canditato, com o número de votos por munícipio somados.

#%%
votacao_candidato_deputado_estadual = pd.read_csv(
    data_path / "brazil_2014_state_deputy_candidates_accepted_votes_with_candidates_info.csv", dtype=data_types
)

#%% [markdown]
# Aqui é interessante comparar alguns números com os obtidos por outra fonte de informação, obtida em
# http://www.tse.jus.br/eleicoes/estatisticas/estatisticas-candidaturas-2014/estatisticas-eleitorais-2014-eleitorado
#

#%%
votacao_candidato_deputado_estadual_tse = pd.read_csv(
    raw_data_path / "tse_votacao/resultado_dep_estadual.csv",
    encoding="ISO-8859-1",
    sep=";",
    na_values="#NULO",
    usecols=["Candidato", "Votação", "Situação"],
)
print("Número de candidatos:", votacao_candidato_deputado_estadual_tse["Candidato"].nunique())
print(
    "Número de candidatos eleitos:",
    votacao_candidato_deputado_estadual_tse["Situação"][
        votacao_candidato_deputado_estadual_tse["Situação"].isin(["Eleito por QP", "Eleito por média"])
    ].count(),
)

#%% [markdown]
# O número obtido, 308, correponde aos 305 encontrados no merge mais os 3 indeferidos.
#%% [markdown]
# ---
#%% [markdown]
# Vamos reunir as informações de perfil e votação, com as informações sobre as doações. Iremos fazer isso através do uso do id_candidate_cpf. Ambmas informações precisam ser do mesmo tipo.

deputados_estaduais_corruptos_brasil = pd.read_csv(
    data_path / "brazil_2014_state_deputy_candidates_donations_total.csv", dtype=data_types
)
#%%
deputados_estaduais_corruptos_brasil.loc[:, "id_candidate_cpf"] = deputados_estaduais_corruptos_brasil[
    "id_candidate_cpf"
]
votacao_candidato_deputado_estadual.loc[:, "id_candidate_cpf"] = votacao_candidato_deputado_estadual["id_candidate_cpf"]
print("Número de candidatos que receberam doações:", deputados_estaduais_corruptos_brasil["id_candidate_cpf"].nunique())
print("Número de candidatos que receberam votos:", votacao_candidato_deputado_estadual["id_candidate_cpf"].nunique())

#%% [markdown]
# Antes de reunir a informação, vamos verificar o que devemos esperar. Para isso criamos uma lista dos CPFs contidos em cada uma delas.

#%%
list_doador = deputados_estaduais_corruptos_brasil["id_candidate_cpf"].unique()
list_votos = votacao_candidato_deputado_estadual["id_candidate_cpf"].unique()

#%% [markdown]
# Primeiro vamos verificar candidatos que receberam doações e mas não constam nem lista de candidatos, nem na lista de votos. Candidato que não receberam votos (nem seu próprio voto?) parecem fazer parte dessa lista.

#%%
nem_nem = deputados_estaduais_corruptos_brasil[
    deputados_estaduais_corruptos_brasil["id_candidate_cpf"].isin(list_votos) == False
]
print("Número de candidatos que estão na lista de doações, mas não nas outras:", nem_nem["id_candidate_cpf"].nunique())
print("Valor:", nem_nem["num_donation_ammount"].sum())
# print("Candidatos:", list(nem_nem['cat_candidate_name'].unique()))

#%% [markdown]
# Optamos por não remover essas doações, pois elas parecem ser extramente suspeitas.

#%%
# deputados_estaduais_corruptos_brasil.drop(nem_nem.index, inplace=True, errors='ignore')

#%% [markdown]
# Vamos verificar quais os candidatos não receberam doações.

#%%
none_donation = votacao_candidato_deputado_estadual[
    votacao_candidato_deputado_estadual["id_candidate_cpf"].isin(list_doador) == False
]
print("Número de candidatos não receberam doações:", none_donation["id_candidate_cpf"].nunique())
