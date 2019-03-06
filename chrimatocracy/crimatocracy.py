

import sys
sys.path.append('../assets')
import benford as bf

import pandas as pd

import matplotlib.pyplot as plt
from matplotlib import ticker

import statsmodels.api as sm
import numpy as np

data_path     = '../data/'
table_path    = '../tables/'
figure_path   = '../figures/'


deputados_estaduais_corruptos_brasil = pd.read_csv(data_path + 'deputados_estaduais_corruptos_brasil.csv')
votos_deputados_estaduais_deferidos  = pd.red_csv(data_path + 'votos_deputados_estaduais_deferidos.csv')
list_doador = deputados_estaduais_corruptos_brasil[u"id_candidate_cpf"].unique()
none_donation = votos_deputados_estaduais_deferidos[votos_deputados_estaduais_deferidos['CPF_CANDIDATO'].isin(list_doador)==False]

#%% [markdown]
# Vamos reunir inserir essa informação de maneira artifical na tabela de doações, apenas para análise do valor da receita
# recebida por esse candidatos, 0.

#%%
none_donation= none_donation.assign(Valor= 0)


#%%
none_donation.columns = ["id_candidate_cpf", 'DESC_SIT_CAND_TOT','SIGLA_UF_votes','NOME_CANDIDATO_votes', "COMPOSICAO_LEGENDA_votes", 'cat_election_state', "TOTAL_VOTOS", 'num_donation_ammount']

#%% [markdown]
# #### Analisaremos os 12961 deputados que receberam doações mais aqueles 2403 que não reberam. 
#%% [markdown]
# Primeiro criamos uma tabela com as doações individuais. Nessa tabela, as informações do candidato estão repetidas, inclusive os votos recebidos.

#%%
merged_receitas = pd.merge(deputados_estaduais_corruptos_brasil, votos_deputados_estaduais_deferidos, left_on='id_candidate_cpf', right_on="CPF_CANDIDATO",how='left', sort=False)
merged_receitas = merged_receitas.append(none_donation)

#%%


#%%
g_merged_receitas = merged_receitas.groupby(['cat_election_state','id_candidate_cpf','SIGLA_UF_votes', 'NOME_CANDIDATO_votes',"COMPOSICAO_LEGENDA_votes","DESC_SIT_CAND_TOT"]).agg({"TOTAL_VOTOS":lambda x: sum(x)/len(x), 'num_donation_ammount':'sum'}).reset_index()
g_merged_receitas.to_csv(data_path + "grouped_donations_brazil__dep_estadual.csv")

del g_merged_receitas
gc.collect()

g_merged_receitas = pd.read_csv(data_path + "grouped_donations_brazil__dep_estadual.csv")

#%%
sns.countplot(x='DESC_SIT_CAND_TOT', data=g_merged_receitas, color = "#1DACD6")
plt.xlabel("Situação")
plt.ylabel("")
plt.grid(False)
plt.tight_layout()
plt.savefig(figure_path + 'situacao_brazil__dep_estadual.png')
#%%

#%% [markdown]
# ### Lei de Benford
#%% [markdown]
# Vamos verificar se a distribuição das receitas adere à lei de Benford.

#%%

all_digits = bf.read_numbers(deputados_estaduais_corruptos_brasil['num_donation_ammount'])       
all_probs = bf.find_probabilities(all_digits)

x2 = bf.find_x2(pd.Series(all_digits))

width = 0.2

indx = np.arange(1, len(all_probs) + 1)
benford = [np.log10(1 + (1.0 / d)) for d in indx]

plt.bar(indx, benford,width, color='r', label="Lei de Benford",)
plt.bar(indx+width , all_probs, width, color='#1DACD6', label=r'Doações($\chi^2$''='+str(round(x2,2))+')')
plt.yscale('log')
plt.xscale('log')
ax = plt.gca()
ax.set_xticks(indx)
ax.set_yticks([0.05,0.1, 0.2, 0.3])
ax.get_xaxis().set_major_formatter(ticker.ScalarFormatter())
ax.get_yaxis().set_major_formatter(ticker.ScalarFormatter())
plt.title("Deputado Estadual")
plt.ylabel("Probabilidade")
plt.grid(False)
plt.legend()
plt.tight_layout()
plt.savefig(figure_path + "benford_distribution_brazil__dep_estadual.png")
plt.show()

#%% [markdown]
# Vamos verificar, utilizando um modelo de regressão logística, se o valor recebido através das doações, é um
# importante preditor para a eleição de um candidato.


params = []
params2 = []
estados = g_merged_receitas['SIGLA_UF_votes'].unique()
final = pd.DataFrame()
for uf in estados:
    X = g_merged_receitas[g_merged_receitas['SIGLA_UF_votes']==uf].set_index("id_candidate_cpf")[['num_donation_ammount','DESC_SIT_CAND_TOT']]
    soma = X['num_donation_ammount'].sum()
    X['num_donation_ammount'] = X['num_donation_ammount'].apply(lambda x: x/soma)
    y=X["DESC_SIT_CAND_TOT"].apply(lambda x: 1 if (x=="ELEITO POR QP" or x=="ELEITO POR MÉDIA") else 0)
    X.drop('DESC_SIT_CAND_TOT', axis=1,inplace=True)
    X_1 = np.append( np.ones((X.shape[0], 1)), X, axis=1)
    logit = sm.Logit(y, X_1)
    result = logit.fit()
    odds = np.exp(float(result.params.values[1]) * 100000/soma)
    sumar = result.summary2()
    llrp = sumar.tables[0][3][5]
    params = pd.DataFrame(sumar.tables[1].loc["x1"].values[0:4], index=["Coeficiente", 'Desvio Padrão','z', "p-valor"]).T
    params["beta"] = r'\beta_1'
    params["LLR p"] = llrp
    params["Odds ratio"] = odds
    params["Estado"] =  uf
    params["Valor"] = g_merged_receitas[g_merged_receitas['SIGLA_UF_votes']==uf]['num_donation_ammount'].sum()
    params["N"] = g_merged_receitas[g_merged_receitas['SIGLA_UF_votes']==uf]["DESC_SIT_CAND_TOT"].count()
    params["n"] = g_merged_receitas[g_merged_receitas['SIGLA_UF_votes']==uf]["DESC_SIT_CAND_TOT"].apply(lambda x: 1 if (x=="ELEITO POR QP" or x=="ELEITO POR MÉDIA") else 0).sum()
    final = final.append(params)
    params2 = pd.DataFrame(sumar.tables[1].loc["const"].values[0:4], index=["Coeficiente", 'Desvio Padrão','z', "p-valor"]).T
    params2["beta"] = r'\beta_0'
    params2["LLR p"] = llrp
    params2["Odds ratio"] = odds
    params2["Estado"] =  uf
    params2["Valor"] = g_merged_receitas[g_merged_receitas['SIGLA_UF_votes']==uf]['num_donation_ammount'].sum()
    params2["N"] = g_merged_receitas[g_merged_receitas['SIGLA_UF_votes']==uf]["DESC_SIT_CAND_TOT"].count()
    params2["n"] = g_merged_receitas[g_merged_receitas['SIGLA_UF_votes']==uf]["DESC_SIT_CAND_TOT"].apply(lambda x: 1 if (x=="ELEITO POR QP" or x=="ELEITO POR MÉDIA") else 0).sum()
    final = final.append(params2)


#%%
fitTab = final.sort_values(by="Estado").set_index("Estado").reset_index()
fitTab = final.set_index(["Estado", "LLR p", "Odds ratio" , "Valor", "N", "n", "beta"])
#print(fitTab.to_latex())
fitTab

#%% [markdown]
# Vamos obter a odds ratio exponenciando o coeficiente obtido. Esse parâmetro nos diz como o incremento ou decremento
# de uma unidade na variável explicativa afeta as chances do candadito ser eleito. Podemos desfazer a normalização adotada para entender a importância do dinheiro para a variável resposta.
#%% [markdown]
# ---

#%%
for j,i in zip(final['Estado'].unique(), final['Odds ratio'].unique()):
    print("Cada R$ 100.000 aumenta as chances de um deputado do ",j," ser eleito em", '%.2f'% float((i-1)*100), "%")
