import pandas as pd

from pathlib import Path

parent_dir = Path().cwd()

data_path     = Path(parent_dir) / 'data/'

data_types = {
    'id_accountant_cnpj':str,
    'id_candidate_cpf':str,
    'id_donator_cpf_cnpj':str,
    'id_original_donator_cpf_cnpj':str,
    'id_donator_effective_cpf_cnpj':str
    }


roles = ["governor", "senator", "federal_deputy", "state_deputy", "district_deputy", "president"]


for role in roles:
        print(f"Generating votes table for {role}.\n")

        receitas_candidatos  = pd.read_csv(data_path / f'brazil_2014_{role}_candidates_donations_aggregated.csv', dtype=data_types, low_memory=False)
        votos_candidatos     = pd.read_csv(data_path / f'brazil_2014_{role}_accepted_candidates_votes_with_candidates_info.csv', dtype=data_types, low_memory=False)
    
        list_doador = receitas_candidatos[u"id_candidate_cpf"].unique()
        none_donation = votos_candidatos[votos_candidatos['id_candidate_cpf'].isin(list_doador)==False]

        #%% [markdown]
        # Vamos reunir inserir essa informação de maneira artifical na tabela de doações, apenas para análise do valor da receita
        # recebida por esse candidatos, 0.


        #%%
        #none_donation.columns = ["id_candidate_cpf", 'DESC_SIT_CAND_TOT','SIGLA_UF_votes','NOME_CANDIDATO_votes', "COMPOSICAO_LEGENDA_votes", 'cat_election_state', "TOTAL_VOTOS", 'num_donation_ammount']

        #%% [markdown]
        # #### Analisaremos os 12961 deputados que receberam doações mais aqueles 2403 que não reberam. 
        #%% [markdown]
        # Primeiro criamos uma tabela com as doações individuais. Nessa tabela, as informações do candidato estão repetidas, inclusive os votos recebidos.

        #%%
        merge_index = ['id_candidate_cpf', 'cat_party', 'cat_federative_unity']
        try:
            merged_receitas = pd.merge(receitas_candidatos, votos_candidatos, left_on=merge_index, right_on=merge_index, how='outer', sort=False)
            merged_receitas['num_donation_ammount'].fillna(0, inplace=True)
            merged_receitas['int_number_of_votes'].fillna(0, inplace=True)
            merged_receitas.to_csv(data_path / f"brazil_2014_{role}_accepted_candidates_votes_with_candidate_info_and_donations.csv", index=False) 
        except:
             print(f"Can't create this file for {role}")