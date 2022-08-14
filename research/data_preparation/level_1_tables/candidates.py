# %%
import glob
import os
import sys
from pathlib import Path

import pandas as pd
import unidecode

parent_dir = Path().cwd()
# %%
raw_data_path = Path(parent_dir) / "raw_data/2014/"
data_path = Path(parent_dir) / "data/"

if not os.path.exists(data_path):
    os.makedirs(data_path)

# %%
print("Creating candidates table...")

translator = {
    "DATA_GERACAO": "date_generation_date",
    "HORA_GERACAO": "hour_generation_hour",
    "ANO_ELEICAO": "int_election_year",
    "NUM_TURNO": "int_election_turn",
    "DESCRICAO_ELEICAO": "cat_election_description",
    "SIGLA_UF": "cat_federative_unity",
    "SIGLA_UE": "cat_state_unity",
    "DESCRICAO_UE": "cat_state_unity_description",
    "CODIGO_CARGO": "id_role_code",
    "DESCRICAO_CARGO": "cat_role_description",
    "NOME_CANDIDATO": "cat_candidate_name",
    "SEQUENCIAL_CANDIDATO": "id_candidate_sequential",
    "NUMERO_CANDIDATO": "id_candidate_number",
    "CPF_CANDIDATO": "id_candidate_cpf",
    "NOME_URNA_CANDIDATO": "cat_candidate_vote_name",
    "COD_SITUACAO_CANDIDATURA": "id_sit_candidate_code",
    "DES_SITUACAO_CANDIDATURA": "cat_sit_candidate_description",
    "NUMERO_PARTIDO": "id_party_number",
    "SIGLA_PARTIDO": "cat_party",
    "NOME_PARTIDO": "cat_party_name",
    "CODIGO_LEGENDA": "id_legend_number",
    "SIGLA_LEGENDA": "cat_legend_number",
    "COMPOSICAO_LEGENDA": "cat_legend_composition",
    "NOME_LEGENDA": "cat_legend_name",
    "CODIGO_OCUPACAO": "id_candidate_occupation",
    "DESCRICAO_OCUPACAO": "cat_candidate_occupation_description",
    "DATA_NASCIMENTO": "date_candidate_birthday",
    "NUM_TITULO_ELEITORAL_CANDIDATO": "id_candidate_electoral_document_number",
    "IDADE_DATA_ELEICAO": "int_candidate_age",
    "CODIGO_SEXO": "id_candidate_sex",
    "DESCRICAO_SEXO": "cat_candidate_sex_description",
    "COD_GRAU_INSTRUCAO": "id_candidate_educational_level",
    "DESCRICAO_GRAU_INSTRUCAO": "cat_candidate_educational_level_description",
    "CODIGO_ESTADO_CIVIL": "id_marriage_status",
    "DESCRICAO_ESTADO_CIVIL": "cat_marriage_status_description",
    "CODIGO_COR_RACA": "id_candidate_race",
    "DESCRICAO_COR_RACA": "cat_candidate_race_description",
    "CODIGO_NACIONALIDADE": "id_candidate_nationality",
    "DESCRICAO_NACIONALIDADE": "cat_candidate_nationality_description",
    "SIGLA_UF_NASCIMENTO": "cat_candidate_origin_state",
    "CODIGO_MUNICIPIO_NASCIMENTO": "id_candidate_origin_municipy_code",
    "NOME_MUNICIPIO_NASCIMENTO": "cat_candidate_origin_municipy_name",
    "DESPESA_MAX_CAMPANHA": "int_campaign_max_spent",
    "COD_SIT_TOT_TURNO": "id_sit_tot_turn_code",
    "DESC_SIT_TOT_TURNO": "cat_sit_tot_turn_description",
    "NM_EMAIL": "cat_email",
}


candidatos_brasil = pd.DataFrame()
path_2014 = raw_data_path / "cadidatos/"
allFiles = glob.glob(os.path.join(path_2014, "consulta_cand_2014_*"))
candidatos_brasil = pd.concat(
    (
        pd.read_csv(
            f,
            header=None,
            names=translator.values(),
            encoding="ISO-8859-1",
            sep=";",
            na_values="#NULO#",
            dtype={"id_candidate_sequential": str},
        )
        for f in allFiles
    )
)
candidatos_brasil.to_csv(data_path / "brazil_2014_candidates.csv", index=False)
