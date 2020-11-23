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
    "pgf.preamble": "\n".join([
         r"\usepackage[utf8x]{inputenc}",
         r"\usepackage[T1]{fontenc}",
         r"\usepackage{cmbright}",
         ])
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




roles = ["governor", "senator", "federal_deputy", "state_deputy", "district_deputy", "president"]

role_donations_table = pd.DataFrame(index=roles, columns=[
                                                    "Tot. (R$)",
                                                    "Num. Doações", 
                                                    "Num. Favorecidos", 
                                                    "Num. Dir. PJ",
                                                    "Tot. Dir. PJ (R$)", 
                                                    "Num. Dir. PF",
                                                    "Tot. Dir. PF (R$)",
                                                    "Num. Ind. PJ", 
                                                    "Tot. Ind. PJ (R$)",
                                                    "Num. Ind. PF",
                                                    "Tot. Ind. PF (R$)"])

for role in roles:
    candidate_donations = pd.read_csv(data_path / f'brazil_2014_{role}_candidates_donations.csv', low_memory=False, dtype=data_types)

    print("Cargos filtrados:", candidate_donations['cat_role_description'].unique())
    
    role_donations_table.loc[role, "Tot. (R$)"] = candidate_donations['num_donation_ammount'].sum()
    print("Total de doações:", candidate_donations['num_donation_ammount'].sum())

    #%%
    role_donations_table.loc[role, "Num. Doações"] = candidate_donations['id_candidate_cpf'].count()
    print("Número total de doações:", candidate_donations['id_candidate_cpf'].count())


    #%%
    role_donations_table.loc[role, "Num. Favorecidos"] = candidate_donations['id_candidate_cpf'].nunique()
    print("Número de candidatos que receberam doações:",candidate_donations['id_candidate_cpf'].nunique())


    #%%
    diretas_pj = candidate_donations[(candidate_donations[u'id_original_donator_cpf_cnpj'].isnull()) 
                            & (candidate_donations[u'id_donator_cpf_cnpj'].astype(str).apply(lambda x: len(x)) == 14)]

    role_donations_table.loc[role, "Num. Dir. PJ"] = diretas_pj["id_candidate_cpf"].count()
    role_donations_table.loc[role, "Tot. Dir. PJ (R$)"] = diretas_pj['num_donation_ammount'].sum()

    print("Número de doações diretas de empresas:",diretas_pj["id_candidate_cpf"].count())
    print("Valor: R$",'%.1e' % diretas_pj['num_donation_ammount'].sum())


    #%%
    diretas_pf = candidate_donations[(candidate_donations[u'id_original_donator_cpf_cnpj'].isnull()) 
                            & (candidate_donations[u'id_donator_cpf_cnpj'].astype(str).apply(lambda x: len(x)) == 11)]

    role_donations_table.loc[role, "Num. Dir. PF"] = diretas_pf["id_candidate_cpf"].count()
    role_donations_table.loc[role, "Tot. Dir. PF (R$)"] = diretas_pf['num_donation_ammount'].sum()
    
    print("Número de doações diretas de indivíduos:",diretas_pf["id_candidate_cpf"].count())
    print("Valor: R$",'%.1e' % diretas_pf['num_donation_ammount'].sum())


    #%%
    indiretas_pj = candidate_donations[(candidate_donations[u'id_original_donator_cpf_cnpj'].astype(str).apply(lambda x: len(x)) == 14) 
                                & (candidate_donations[u'id_donator_cpf_cnpj'].astype(str).apply(lambda x: len(x)) == 14)]

    role_donations_table.loc[role, "Num. Ind. PJ"] = indiretas_pj["id_candidate_cpf"].count()
    role_donations_table.loc[role, "Tot. Ind. PJ (R$)"] = indiretas_pj['num_donation_ammount'].sum()
    
    print("Número de doações indiretas provenientes de empresas:",indiretas_pj["id_candidate_cpf"].count())
    print("Valor: R$",'%.1e' % indiretas_pj['num_donation_ammount'].sum())


    #%%
    indiretas_pf = candidate_donations[(candidate_donations[u'id_original_donator_cpf_cnpj'].astype(str).apply(lambda x: len(x)) == 11) 
                                & (candidate_donations[u'id_donator_cpf_cnpj'].astype(str).apply(lambda x: len(x)) == 14)]

    role_donations_table.loc[role, "Num. Ind. PF"] = indiretas_pf["id_candidate_cpf"].count()
    role_donations_table.loc[role, "Tot. Ind. PF (R$)"] = indiretas_pf['num_donation_ammount'].sum()

    print("Número de doações indiretas provenientes de individuos:",indiretas_pf["id_candidate_cpf"].count())
    print("Valor: R$",'%.1e' % indiretas_pf['num_donation_ammount'].sum())


    #%%
    # indiretas_teste = candidate_donations[(candidate_donations[u'id_original_donator_cpf_cnpj'].isnull() == False) 
    #                                 & (candidate_donations[u'id_donator_cpf_cnpj'].astype(str).apply(lambda x: len(x)) == 11)]

    # print(indiretas_teste)
    # print("Número de doações com intermediário sendo uma pessoa física",indiretas_teste["id_candidate_cpf"].count())
    # print("Valor: R$",'%.1e' % indiretas_teste['num_donation_ammount'].sum())
    # if indiretas_teste["id_candidate_cpf"].count() == 0 :print("UFA!")


    # #%%
    # unknow = candidate_donations[(candidate_donations[u'cat_original_donator_name'].isnull()) 
    #                     & (candidate_donations[u'id_donator_cpf_cnpj'].isnull())]
    # #candidate_donations.drop(unknow.index, inplace=True)

    # print("Número de doações não rastreadas",unknow["id_candidate_cpf"].count())
    # print("Valor: R$",'%.1e' % unknow['num_donation_ammount'].sum())

    #%% [markdown]
    # ----
    #%% [markdown]
    # Estatística básica das doações

    #%%
    stats_donations = pd.DataFrame(candidate_donations['num_donation_ammount'].describe()).T.append(pd.DataFrame(candidate_donations[candidate_donations[ 'id_donator_effective_cpf_cnpj'].astype(str).apply(lambda x: len(x)) == 14]['num_donation_ammount'].describe()).T.append(pd.DataFrame(candidate_donations[candidate_donations[ 'id_donator_effective_cpf_cnpj'].astype(str).apply(lambda x: len(x)) == 11]['num_donation_ammount'].describe()).T))
    stats_donations.index = ['Todos','CNPJ','CPF']
    stats_donations.columns = ['N', 'Mean', 'Std', 'Min', '25%', "50%", '75%', 'Max']
    print("Estatistica das Doações:\n", stats_donations)
    with open(table_path / f'brazil_2014_{role}_donations_statistics.tex', 'w') as tf:
        tf.write(stats_donations.to_latex(escape=True, bold_rows=True, formatters={
    'N': '{:.0f}'.format,
    'Mean': '{:.2f}R$'.format,
    'Std': '{:.2f}R$'.format,
    'Min': '{:.2f}R$'.format,
    '25%': '{:.2f}R$'.format,
    '50%': '{:.2f}R$'.format,
    '75%': '{:.2f}R$'.format,
    'Max': '{:.2f}R$'.format,
    }))


    #%% [markdown]
    # ## Distribuição aculumada das doações

    #%%
    import numpy as np
    import matplotlib.pyplot as plt

    # method 1
    H2,X2 = np.histogram( np.log10(candidate_donations['num_donation_ammount']), density=True)
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
    plt.savefig(figure_path / f'brazil_2014_{role}_donations_cdf__cpf_cnpj.pgf')


    #%%
    import numpy as np
    import matplotlib.pyplot as plt

    # method 1
    H2,X2 = np.histogram( np.log10(candidate_donations[candidate_donations[ 'id_donator_effective_cpf_cnpj'].astype(str).apply(lambda x: len(x)) == 14]['num_donation_ammount']),  density=True )
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
    plt.savefig(figure_path / f'brazil_2014_{role}_donations_cdf__cnpj.pgf')


    #%%
    import numpy as np
    import matplotlib.pyplot as plt

    # method 1
    H2,X2 = np.histogram( np.log10(candidate_donations[candidate_donations[ 'id_donator_effective_cpf_cnpj'].astype(str).apply(lambda x: len(x)) == 11]['num_donation_ammount']), density=True )
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
    plt.savefig(figure_path / f'brazil_2014_{role}_donations_cdf__cpf.pgf')

    #%% [markdown]
    # Somamos os valores provenientes de um mesmo doador para um dado candidato (mais de uma doação ou repasse de doações oriundas desse doador)

    #%%
    g_candidate_donations = pd.read_csv(data_path / f'brazil_2014_{role}_candidates_donations_aggregated.csv', dtype=data_types, low_memory=False)

    #%%
    print("Número de empresas que doaram, direta ou indiretamente, para candidatos",g_candidate_donations[g_candidate_donations[ 'id_donator_effective_cpf_cnpj'].astype(str).apply(lambda x: len(x))==14]['num_donation_ammount'].count())
    print("Valor: R$",'%.1e' % g_candidate_donations[g_candidate_donations[ 'id_donator_effective_cpf_cnpj'].astype(str).apply(lambda x: len(x))==14]['num_donation_ammount'].sum())


    #%%
    print("Número de indivíduos que doaram, direta ou indiretamente, para candidatos",g_candidate_donations[g_candidate_donations[ 'id_donator_effective_cpf_cnpj'].astype(str).apply(lambda x: len(x))==11]['num_donation_ammount'].count())
    print("Valor: R$",'%.1e' % g_candidate_donations[g_candidate_donations[ 'id_donator_effective_cpf_cnpj'].astype(str).apply(lambda x: len(x))==11]['num_donation_ammount'].sum())

    #%% [markdown]
    # Box-plot do valor das doações em escala logaritmica, feitas para candidatos, agregadas por partido. Os valores abragem aproximadamente 7 escalas de grandeza mas possuem valores médios similares.

    #%%
    g = sns.boxplot(x='cat_federative_unity', y='num_donation_ammount', data=g_candidate_donations, color = '#1DACD6')
    plt.yscale('log')
    plt.xticks(rotation=90)
    plt.grid(False)
    plt.tight_layout()
    plt.savefig(figure_path / f"brazil_2014_{role}_candidates_donations_by_party_boxplot.pgf")

with open(table_path / f'brazil_2014_donations_numbers_by_role.tex', 'w') as tf:
        tf.write(role_donations_table.to_latex(index=True, escape=True,bold_rows=True ,formatters={
    'Tot. (R$)': '{:.2e}'.format,
    'Num. Doações': '{:.0f}'.format,
    'Num. Favorecidos': '{:.0f}'.format,
    'Num. Dir. PJ': '{:.0f}'.format,
    'Tot. Dir. PJ (R$)': '{:.2e}'.format,
    'Num. Dir. PF': '{:.0f}'.format,
    'Tot. Dir. PF (R$)': '{:.2e}'.format,
    'Num. Ind. PJ': '{:.0f}'.format,   
    'Tot. Ind. PJ (R$)': '{:.2e}'.format,
    'Num. Ind. PF': '{:.0f}'.format,
    'Tot. Ind. PF (R$)': '{:.2e}'.format
    }))

