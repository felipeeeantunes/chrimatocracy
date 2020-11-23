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

# %%
print('Creating votes table...')

translator = {
    'DATA_GERACAO': 'date_generation_date',
    'HORA_GERACAO': 'hour_generation_hour',
    'ANO_ELEICAO': 'int_election_year',
    'NUM_TURNO': 'int_election_turn',
    'DESCRICAO_ELEICAO': 'cat_election_description',
    'SIGLA_UF': 'cat_federative_unity',
    'SIGLA_UE': 'cat_state_unity',
    'CODIGO_MUNICIPIO': 'id_municipy_code',
    'NOME_MUNICIPIO': 'cat_municipy_name',
    'NUMERO_ZONA': 'id_electoral_zone_number',
    'CODIGO_CARGO': 'id_role_code',
    'NUMERO_CAND': 'id_candidate_number',
    'SQ_CANDIDATO': 'id_candidate_sequential',
    'NOME_CANDIDATO': 'cat_candidate_name',
    'NOME_URNA_CANDIDATO': 'cat_candidate_vote_name',
    'DESCRICAO_CARGO': 'cat_role_description',
    'COD_SIT_CAND_SUPERIOR': 'int_sit_candidate_superior_code',
    'DESC_SIT_CAND_SUPERIOR': 'cat_sit_candidate_superior_description',
    'CODIGO_SIT_CANDIDATO': 'id_sit_candidate_code',
    'DESC_SIT_CANDIDATO' : 'cat_sit_candidate',
    'CODIGO_SIT_CAND_TOT': 'id_sit_candidate_tot_code',       
    'DESC_SIT_CAND_TOT':'cat_sit_candidate_tot_description', 
    'NUMERO_PARTIDO': 'id_party_number', 
    'SIGLA_PARTIDO': 'cat_party', 
    'NOME_PARTIDO': 'cat_party_name',       
    'SEQUENCIAL_LEGENDA': 'id_legend_sequential', 
    'NOME_COLIGACAO': 'cat_coligation_name', 
    'COMPOSICAO_LEGENDA': 'cat_legend_composition',       
    'TOTAL_VOTOS': 'int_number_of_votes', 
    'TRANSITO': 'cat_transit'
}



votos_brasil = pd.DataFrame()
path_2014 = raw_data_path / 'votacao/'
allFiles = glob.glob(os.path.join(
    path_2014, "votacao_candidato_munzona_2014_*"))
votos_brasil = pd.concat((pd.read_csv(f, header=None, names=translator.values(),  encoding="ISO-8859-1",
                                      sep=';', na_values='#NULO#', dtype={'id_candidate_sequential': str}) for f in allFiles))
votos_brasil.to_csv(
    data_path / 'brazil_2014_votes.csv', index=False)