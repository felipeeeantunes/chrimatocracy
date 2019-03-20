#%% [markdown]
# # Investigação das doações para Deputado Estadual no Brasil
# 

#%%
import sys
import gc
import os
from pathlib import Path

parent_dir = Path().cwd()
print((Path(parent_dir) / 'assets/'))
sys.path.append(str(Path(parent_dir) / 'assets/'))
import benford as bf

import pandas as pd
import seaborn as sns
import numpy as np

pd.options.display.float_format = '{:.2f}'.format

#%%
raw_data_path = Path(parent_dir) / 'raw_data/2014/'
data_path     = Path(parent_dir) / 'data/'
table_path    = Path(parent_dir) / 'tables/'
figure_path   = Path(parent_dir) / 'figures/'

directories = [table_path, figure_path]

for directory in directories:
    if not os.path.exists(directory):
        os.makedirs(directory)


#%%
import matplotlib.pyplot as plt
import matplotlib as mpl
mpl.use("pgf")
pgf_with_pdflatex = {
    "pgf.texsystem": "pdflatex",
    "pgf.preamble": [
         r"\usepackage[utf8x]{inputenc}",
         r"\usepackage[T1]{fontenc}",
         r"\usepackage{cmbright}",
         ]
}
mpl.rcParams.update(pgf_with_pdflatex)
#get_ipython().run_line_magic('matplotlib', 'inline')

rc={'savefig.dpi': 75, 'figure.autolayout': False, 'figure.figsize': [12, 8], 'axes.labelsize': 18,   'axes.titlesize': 18, 'font.size': 18, 'lines.linewidth': 2.0, 'lines.markersize': 8, 'legend.fontsize': 16,   'xtick.labelsize': 16, 'ytick.labelsize': 16}

sns.set(style='whitegrid',rc=rc)

data_types = {
    'id_accountant_cnpj':str,
    'id_candidate_cpf':str,
    'id_donator_cpf_cnpj':str,
    'id_original_donator_cpf_cnpj':str,
    'id_donator_effective_cpf_cnpj':str
    }

comites_brasil    = pd.read_csv(data_path / "receitas_comites_2014_brasil_english.csv", dtype = data_types)
partidos_brasil   = pd.read_csv(data_path / "receitas_partidos_2014_brasil_english.csv", dtype = data_types)
corruptos_brasil  = pd.read_csv(data_path / "receitas_candidatos_2014_brasil_english.csv", dtype = data_types)

#%% [markdown]
# #### Partidos

#%%
print("Número total de doações a partidos:", partidos_brasil['cat_party'].count())


#%%
print("Número de partidos que receberam doações:", partidos_brasil['cat_party'].nunique())


#%%
diretas_part_pj = partidos_brasil[(partidos_brasil[u'id_original_donator_cpf_cnpj'].isnull()) & (partidos_brasil[u'id_donator_cpf_cnpj'].str.len() == 14)]

print("Número de doações diretas de empresas a partidos:",diretas_part_pj['cat_party'].count())
print("Valor: R$",'%.1e' % diretas_part_pj['num_donation_ammount'].sum())


#%%
diretas_part_pf = partidos_brasil[(partidos_brasil[u'id_original_donator_cpf_cnpj'].isnull()) & (partidos_brasil[u'id_donator_cpf_cnpj'].str.len() == 11)]

print("Número de doações diretas de indivíduos a partidos:",diretas_part_pf['cat_party'].count())
print("Valor: R$",'%.1e' % diretas_part_pf['num_donation_ammount'].sum())


#%%
indiretas_part_pj = partidos_brasil[(partidos_brasil[u'id_original_donator_cpf_cnpj'].str.len() == 14) & (partidos_brasil[u'id_donator_cpf_cnpj'].str.len() == 14)]


print("Número de doações indiretas provenientes de empresas a partidos:",indiretas_part_pj['cat_party'].count())
print("Valor: R$",'%.1e' % indiretas_part_pj['num_donation_ammount'].sum())


#%%
indiretas_part_pf = partidos_brasil[(partidos_brasil[u'id_original_donator_cpf_cnpj'].str.len() == 11) & (partidos_brasil[u'id_donator_cpf_cnpj'].str.len() == 14)]

print("Número de doações indiretas provenientes de individuos a partidos:",indiretas_part_pf['cat_party'].count())
print("Valor: R$",'%.1e' % indiretas_part_pf['num_donation_ammount'].sum())


#%%
unknow_part = partidos_brasil[(partidos_brasil[u'cat_original_donator_name'].isnull()) & (partidos_brasil[u'id_donator_cpf_cnpj'].isnull())]
#deputados_estaduais_corruptos_brasil.drop(unknow.index, inplace=True)

print("Número de doações não rastreadas a partidos",unknow_part['cat_party'].count())
print("Valor: R$",'%.1e' % unknow_part['num_donation_ammount'].sum())

#%% [markdown]
# #### Comitês

#%%
print(comites_brasil['cat_commitee_type'].unique())


#%%
print("Número total de doações a comites:", comites_brasil['id_commitee_seq'].count())


#%%
print("Número de comites que receberam doações:",comites_brasil['id_commitee_seq'].nunique())


#%%
diretas_com_pj = comites_brasil[(comites_brasil[u'id_original_donator_cpf_cnpj'].isnull()) & (comites_brasil[u'id_donator_cpf_cnpj'].str.len() == 14)]

print("Número de doações diretas de empresas a comites:",diretas_com_pj['id_commitee_seq'].count())
print("Valor: R$",'%.1e' % diretas_com_pj['num_donation_ammount'].sum())


#%%
diretas_com_pf = comites_brasil[(comites_brasil[u'id_original_donator_cpf_cnpj'].isnull()) & (comites_brasil[u'id_donator_cpf_cnpj'].str.len() == 11)]

print("Número de doações diretas de indivíduos a comites:",diretas_com_pf['id_commitee_seq'].count())
print("Valor: R$",'%.1e' % diretas_com_pf['num_donation_ammount'].sum())


#%%
indiretas_com_pj = comites_brasil[(comites_brasil[u'id_original_donator_cpf_cnpj'].str.len() == 14) 
                                  & (comites_brasil[u'id_donator_cpf_cnpj'].str.len() == 14)]


print("Número de doações indiretas provenientes de empresas a comites:",indiretas_com_pj['id_commitee_seq'].count())
print("Valor: R$",'%.1e' % indiretas_com_pj['num_donation_ammount'].sum())


#%%
indiretas_com_pf = comites_brasil[(comites_brasil[u'id_original_donator_cpf_cnpj'].str.len() == 11) 
                                  & (comites_brasil[u'id_donator_cpf_cnpj'].str.len() == 14)]

print("Número de doações indiretas provenientes de individuos a comites:",indiretas_com_pf['id_commitee_seq'].count())
print("Valor: R$",'%.1e' % indiretas_com_pf['num_donation_ammount'].sum())


#%%
unknow_com = comites_brasil[(comites_brasil[u'cat_original_donator_name'].isnull()) 
                            & (comites_brasil[u'id_donator_cpf_cnpj'].isnull())]
#deputados_estaduais_corruptos_brasil.drop(unknow.index, inplace=True)

print("Número de doações não rastreadas a comites",unknow_com['id_commitee_seq'].count())
print("Valor: R$",'%.1e' % unknow_com['num_donation_ammount'].sum())

#%% [markdown]
# #### Candidatos a Deputado Estadual

#%%
deputados_estaduais_corruptos_brasil = corruptos_brasil[(corruptos_brasil['cat_political_office'] == "Deputado Estadual")]
deputados_estaduais_corruptos_brasil.to_csv(data_path / 'deputados_estaduais_corruptos_brasil.csv', index=False)

print("Cargos filtrados:", deputados_estaduais_corruptos_brasil['cat_political_office'].unique())


#%%
print("Número total de doações:", deputados_estaduais_corruptos_brasil['id_candidate_cpf'].count())


#%%
print("Número de candidatos que receberam doações:",deputados_estaduais_corruptos_brasil['id_candidate_cpf'].nunique())


#%%
diretas_pj = deputados_estaduais_corruptos_brasil[(deputados_estaduais_corruptos_brasil[u'id_original_donator_cpf_cnpj'].isnull()) 
                           & (deputados_estaduais_corruptos_brasil[u'id_donator_cpf_cnpj'].str.len() == 14)]

print("Número de doações diretas de empresas:",diretas_pj["id_candidate_cpf"].count())
print("Valor: R$",'%.1e' % diretas_pj['num_donation_ammount'].sum())


#%%
diretas_pf = deputados_estaduais_corruptos_brasil[(deputados_estaduais_corruptos_brasil[u'id_original_donator_cpf_cnpj'].isnull()) 
                           & (deputados_estaduais_corruptos_brasil[u'id_donator_cpf_cnpj'].str.len() == 11)]

print("Número de doações diretas de indivíduos:",diretas_pf["id_candidate_cpf"].count())
print("Valor: R$",'%.1e' % diretas_pf['num_donation_ammount'].sum())


#%%
indiretas_pj = deputados_estaduais_corruptos_brasil[(deputados_estaduais_corruptos_brasil[u'id_original_donator_cpf_cnpj'].str.len() == 14) 
                             & (deputados_estaduais_corruptos_brasil[u'id_donator_cpf_cnpj'].str.len() == 14)]

print("Número de doações indiretas provenientes de empresas:",indiretas_pj["id_candidate_cpf"].count())
print("Valor: R$",'%.1e' % indiretas_pj['num_donation_ammount'].sum())


#%%
indiretas_pf = deputados_estaduais_corruptos_brasil[(deputados_estaduais_corruptos_brasil[u'id_original_donator_cpf_cnpj'].str.len() == 11) 
                             & (deputados_estaduais_corruptos_brasil[u'id_donator_cpf_cnpj'].str.len() == 14)]

print("Número de doações indiretas provenientes de individuos:",indiretas_pf["id_candidate_cpf"].count())
print("Valor: R$",'%.1e' % indiretas_pf['num_donation_ammount'].sum())


#%%
indiretas_teste = deputados_estaduais_corruptos_brasil[(deputados_estaduais_corruptos_brasil[u'id_original_donator_cpf_cnpj'].isnull() == False) 
                                & (deputados_estaduais_corruptos_brasil[u'id_donator_cpf_cnpj'].str.len() == 11)]


print("Número de doações com intermediário sendo uma pessoa física",indiretas_teste["id_candidate_cpf"].count())
print("Valor: R$",'%.1e' % indiretas_teste['num_donation_ammount'].sum())
if indiretas_teste["id_candidate_cpf"].count() == 0 :print("UFA!")


#%%
unknow = deputados_estaduais_corruptos_brasil[(deputados_estaduais_corruptos_brasil[u'cat_original_donator_name'].isnull()) 
                       & (deputados_estaduais_corruptos_brasil[u'id_donator_cpf_cnpj'].isnull())]
#deputados_estaduais_corruptos_brasil.drop(unknow.index, inplace=True)

print("Número de doações não rastreadas",unknow["id_candidate_cpf"].count())
print("Valor: R$",'%.1e' % unknow['num_donation_ammount'].sum())

#%% [markdown]
# ----
#%% [markdown]
# Estatística básica das doações

#%%
stats_donations = pd.DataFrame(deputados_estaduais_corruptos_brasil['num_donation_ammount'].describe()).T.append(pd.DataFrame(deputados_estaduais_corruptos_brasil[deputados_estaduais_corruptos_brasil[ 'id_donator_effective_cpf_cnpj'].str.len() == 14]['num_donation_ammount'].describe()).T.append(pd.DataFrame(deputados_estaduais_corruptos_brasil[deputados_estaduais_corruptos_brasil[ 'id_donator_effective_cpf_cnpj'].str.len() == 11]['num_donation_ammount'].describe()).T))
stats_donations.index = ['Todos','CNPJ','CPF']
stats_donations.columns = ['N', 'Mean', 'Std', 'Min', '25%', "50%", '75%', 'Max']
print("Estatistica das Doações:\n", stats_donations)
with open(table_path / 'deputado_estadual_donation_statistics__brasil_.tex', 'w') as tf:
     tf.write(stats_donations.to_latex())


#%% [markdown]
# ## Distribuição aculumada das doações

#%%
import numpy as np
import matplotlib.pyplot as plt

# method 1
H2,X2 = np.histogram( np.log10(deputados_estaduais_corruptos_brasil['num_donation_ammount']), density=True)
dx2 = X2[1] - X2[0]
F2 = np.cumsum(H2)*dx2
plt.fill_between(X2[1:], F2,facecolor="#1DACD6", alpha=.7)
plt.plot(X2[1:], F2, c='#1DACD6', linestyle='-')
plt.ylabel("CDF")
plt.xlabel('ln(Valor(R$))')
plt.title("Todos")
#plt.legend()
plt.grid(False)
plt.tight_layout()
plt.savefig(figure_path / 'cdf_brazil_donations__dep_estadual_cpf_cnpj.pgf')


#%%
import numpy as np
import matplotlib.pyplot as plt

# method 1
H2,X2 = np.histogram( np.log10(deputados_estaduais_corruptos_brasil[deputados_estaduais_corruptos_brasil[ 'id_donator_effective_cpf_cnpj'].str.len() == 14]['num_donation_ammount']),  density=True )
dx2 = X2[1] - X2[0]
F2 = np.cumsum(H2)*dx2
plt.plot(X2[1:], F2, c='#1DACD6', linestyle='-')
plt.fill_between(X2[1:], F2,facecolor="#1DACD6", alpha=.7)
plt.ylabel("CDF")
plt.xlabel('ln(Valor(R$))')
plt.title("Pessoa Jurídica")
#plt.legend()
plt.grid(False)
plt.tight_layout()
plt.savefig(figure_path / 'cdf_brazil_donations___dep_estadual_cnpj.pgf')


#%%
import numpy as np
import matplotlib.pyplot as plt

# method 1
H2,X2 = np.histogram( np.log10(deputados_estaduais_corruptos_brasil[deputados_estaduais_corruptos_brasil[ 'id_donator_effective_cpf_cnpj'].str.len() == 11]['num_donation_ammount']), density=True )
dx2 = X2[1] - X2[0]
F2 = np.cumsum(H2)*dx2
plt.plot(X2[1:], F2, c='#1DACD6', linestyle='-')
plt.fill_between(X2[1:], F2,facecolor="#1DACD6", alpha=.7)
plt.ylabel("CDF")
plt.xlabel('ln(Valor(R$))')
plt.title("Pessoa Física")
#plt.legend()
plt.grid(False)
plt.tight_layout()
plt.savefig(figure_path / 'cdf_brazil_donations___dep_estadual_cpf.pgf')

#%% [markdown]
# Somamos os valores provenientes de um mesmo doador para um dado candidato (mais de uma doação ou repasse de doações oriundas desse doador)

#%%
g_deputados_estaduais_corruptos_brasil = deputados_estaduais_corruptos_brasil.groupby([u'id_candidate_cpf', 'id_donator_effective_cpf_cnpj','cat_party', 'cat_political_office', 'cat_election_state']).agg({'num_donation_ammount': 'sum'})
g_deputados_estaduais_corruptos_brasil = pd.DataFrame(g_deputados_estaduais_corruptos_brasil)
g_deputados_estaduais_corruptos_brasil = g_deputados_estaduais_corruptos_brasil.reset_index()
g_deputados_estaduais_corruptos_brasil.to_csv(data_path / 'total_agrupado_deputados_estaduais_corruptos_brasil.csv', index=False)

#%%
print("Número de empresas que doaram, direta ou indiretamente, para candidatos",g_deputados_estaduais_corruptos_brasil[g_deputados_estaduais_corruptos_brasil[ 'id_donator_effective_cpf_cnpj'].str.len()==14]['num_donation_ammount'].count())
print("Valor: R$",'%.1e' % g_deputados_estaduais_corruptos_brasil[g_deputados_estaduais_corruptos_brasil[ 'id_donator_effective_cpf_cnpj'].str.len()==14]['num_donation_ammount'].sum())


#%%
print("Número de indivíduos que doaram, direta ou indiretamente, para candidatos",g_deputados_estaduais_corruptos_brasil[g_deputados_estaduais_corruptos_brasil[ 'id_donator_effective_cpf_cnpj'].str.len()==11]['num_donation_ammount'].count())
print("Valor: R$",'%.1e' % g_deputados_estaduais_corruptos_brasil[g_deputados_estaduais_corruptos_brasil[ 'id_donator_effective_cpf_cnpj'].str.len()==11]['num_donation_ammount'].sum())

#%% [markdown]
# Box-plot do valor das doações em escala logaritmica, feitas para candidatos, agregadas por partido. Os valores abragem aproximadamente 7 escalas de grandeza mas possuem valores médios similares.

#%%
g = sns.boxplot(x='cat_election_state', y='num_donation_ammount', data=g_deputados_estaduais_corruptos_brasil, color = '#1DACD6')
plt.yscale('log')
plt.xticks(rotation=90)
plt.grid(False)
plt.tight_layout()
plt.savefig(figure_path / "boxplot_donations__dep_estadual_by_party.pgf")

#%% [markdown]
# ##  Informações sobre a votação e o perfil dos candidatos
#%% [markdown]
# A tabela de candidatos possui informações sobre o perfil dos candidatos,tais como raça, escolaridade, estado civil,
# além das informações relacionadas às eleições.  Além disso, para associarmos a lista de votação à lista de doações, precisamos da informação sobre o CPF, que só consta na tabela de candidatos. Esse mapeamento somente pode ser feito através da sequencial, constante tanto na tabela de candidatos quando na de votos. 

candidatos_brasil = pd.read_csv(data_path / 'consulta_cand_2014_full.csv')
#%% [markdown]
# Filtramos candidatos para os cargos estudos, cuja candidatura foi deferida

#%%
candidatos_deputado_estadual_indeferidos = pd.read_csv(data_path / 'candidatos_deputado_estadual_indeferidos.csv')
print("Cargos:", candidatos_deputado_estadual_indeferidos["DESCRICAO_CARGO"].unique())
print("Número de canidatos indeferidos",candidatos_deputado_estadual_indeferidos["SEQUENCIAL_CANDIDATO"].nunique())     


#%%
candidatos_deputado_estadual_deferidos = pd.read_csv(data_path / 'candidatos_deputado_estadual_deferidos.csv')
print("Cargos:", candidatos_deputado_estadual_deferidos["DESCRICAO_CARGO"].unique())
print("Número de candidatos deferidos",candidatos_deputado_estadual_deferidos["SEQUENCIAL_CANDIDATO"].nunique())     


#%% [markdown]
# Criamos a lista de candidatos baseado na sua sequencial, utilizada para cruzar essa base de informações com a base votação. 

#%%
list_cand = candidatos_deputado_estadual_deferidos["SEQUENCIAL_CANDIDATO"].unique()
print("Número de candidatos na lista:",len(list_cand))  

#%% [markdown]
# A tabela de votos possui informação sobre os dados apurados da votação, no primeiro e segundo turno, em cada munício, além de informações eleitorias.
votos_brasil = pd.read_csv(data_path / 'votacao_candidato_munzona_2014_full.csv')

#%%
votos_deputados_estaduais_indeferidos = pd.read_csv(data_path / 'votos_deputados_estaduais_indeferidos.csv')
print("Cargos:", votos_deputados_estaduais_indeferidos["DESCRICAO_CARGO"].unique())
print("Número de candidatos indeferidos",votos_deputados_estaduais_indeferidos["SQ_CANDIDATO"].nunique())                       


#%%
votos_deputados_estaduais_deferidos = pd.read_csv(data_path / 'votos_deputados_estaduais_deferidos.csv')
print("Cargos:", votos_deputados_estaduais_deferidos["DESCRICAO_CARGO"].unique())
print("Número de candidatos deferidos",votos_deputados_estaduais_deferidos["SQ_CANDIDATO"].nunique()) 


#%% [markdown]
# Percebemos que há um candidato a mais do que o da lista de candidatos. É um fato curioso um candidato receber votos, mas não constar na lista inicial.
# Trata-se de Tâmara Joana Biolo Soares, que não foi eleita.

#%% [markdown]
# Vamos user uma tabela contendo informações do canditato, com o número de votos por munícipio somados.

#%%
votacao_candidato_deputado_estadual = pd.read_csv(data_path / "votacao_candidato_deputado_estadual_deferidos_soma_municipios.csv")

#%% [markdown]
# Aqui é interessante comparar alguns números com os obtidos por outra fonte de informação, obtida em 
# http://www.tse.jus.br/eleicoes/estatisticas/estatisticas-candidaturas-2014/estatisticas-eleitorais-2014-eleitorado
# 

#%%
votacao_candidato_deputado_estadual_tse = pd.read_csv(raw_data_path / 'tse_votacao/resultado_dep_estadual.csv', encoding="ISO-8859-1", sep=';', na_values = '#NULO', usecols  = ["Candidato", "Votação", "Situação"])
print("Número de candidatos:",votacao_candidato_deputado_estadual_tse["Candidato"].nunique()) 
print("Número de candidatos eleitos:",  votacao_candidato_deputado_estadual_tse["Situação"][votacao_candidato_deputado_estadual_tse["Situação"].isin(["Eleito por QP", "Eleito por média"])].count())

#%% [markdown]
# O número obtido, 308, correponde aos 305 encontrados no merge mais os 3 indeferidos.
#%% [markdown]
# ---
#%% [markdown]
# Vamos reunir as informações de perfil e votação, com as informações sobre as doações. Iremos fazer isso através do uso do id_candidate_cpf. Ambmas informações precisam ser do mesmo tipo.

#%%
deputados_estaduais_corruptos_brasil.loc[:,'id_candidate_cpf'] = deputados_estaduais_corruptos_brasil['id_candidate_cpf'].apply(lambda x: int(x))
votacao_candidato_deputado_estadual.loc[:,'CPF_CANDIDATO'] = votacao_candidato_deputado_estadual['CPF_CANDIDATO'].apply(lambda x: int(x))
print("Número de candidatos que receberam doações:",deputados_estaduais_corruptos_brasil['id_candidate_cpf'].nunique())
print("Número de candidatos que receberam votos:",votacao_candidato_deputado_estadual['CPF_CANDIDATO'].nunique())

#%% [markdown]
# Antes de reunir a informação, vamos verificar o que devemos esperar. Para isso criamos uma lista dos CPFs contidos em cada uma delas.

#%%
list_doador = deputados_estaduais_corruptos_brasil[u"id_candidate_cpf"].unique()
list_votos  = votacao_candidato_deputado_estadual['CPF_CANDIDATO'].unique()

#%% [markdown]
# Primeiro vamos verificar candidatos que receberam doações e mas não constam nem lista de candidatos, nem na lista de votos. Candidato que não receberam votos (nem seu próprio voto?) parecem fazer parte dessa lista.

#%%
nem_nem = deputados_estaduais_corruptos_brasil[deputados_estaduais_corruptos_brasil['id_candidate_cpf'].isin(list_votos)==False]
print("Número de candidatos que estão na lista de doações, mas não nas outras:", nem_nem['id_candidate_cpf'].nunique())
print("Valor:",nem_nem['num_donation_ammount'].sum())
#print("Candidatos:", list(nem_nem['cat_candidate_name'].unique()))

#%% [markdown]
# Optamos por não remover essas doações, pois elas parecem ser extramente suspeitas.

#%%
#deputados_estaduais_corruptos_brasil.drop(nem_nem.index, inplace=True, errors='ignore')

#%% [markdown]
# Vamos verificar quais os candidatos não receberam doações.

#%%
none_donation = votacao_candidato_deputado_estadual[votacao_candidato_deputado_estadual['CPF_CANDIDATO'].isin(list_doador)==False]
print("Número de candidatos não receberam doações:", none_donation['CPF_CANDIDATO'].nunique())
