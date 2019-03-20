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
data_path     = Path(parent_dir) / 'data/'

if not os.path.exists(data_path):
    os.makedirs(data_path)


# %%
print('Reading and decoding files...')
comites_brasil = pd.read_csv(raw_data_path / "receitas_comites_2014_brasil.txt",
                             encoding="ISO-8859-1", sep=';', na_values='#NULO', dtype=str)
partidos_brasil = pd.read_csv(raw_data_path / "receitas_partidos_2014_brasil.txt",
                              encoding="ISO-8859-1", sep=';', na_values='#NULO', dtype=str)
corruptos_brasil = pd.read_csv(raw_data_path / "receitas_candidatos_2014_brasil.txt",
                               encoding="ISO-8859-1", sep=';', na_values='#NULO', dtype=str)
print('Done.')

# %% [markdown]
# ## Cria dicionario com colunnas em inglês
# %%
translator = {
    'CNPJ Prestador Conta': 'id_accountant_cnpj',
    'CPF do candidato': 'id_candidate_cpf',
    'CPF/CNPJ do doador': 'id_donator_cpf_cnpj',
    'CPF/CNPJ do doador originário': 'id_original_donator_cpf_cnpj',
    'Cargo': 'cat_political_office',
    'Cod setor econômico do doador': 'id_donator_economic_sector',
    'Cód. Eleição': 'id_election',
    'Data da receita': 'dat_donation_date',
    'Data e hora': 'dat_donation_date_time',
    'Desc. Eleição': 'cat_election_description',
    'Descricao da receita': 'cat_donation_description',
    'Especie recurso': 'cat_donation_method',
    'Fonte recurso': 'cat_donation_source',
    'Nome candidato': 'cat_candidate_name',
    'Nome do doador': 'cat_donator_name',
    'Nome do doador (Receita Federal)': 'cat_donator_name2',
    'Nome do doador originário': 'cat_original_donator_name',
    'Nome do doador originário (Receita Federal)': 'cat_original_donator_name2',
    'Numero Recibo Eleitoral': 'id_receipt_num',
    'Numero candidato': 'id_candidate_num',
    'Numero do documento': 'id_document_num',
    'Número candidato doador': 'id_donator_number',
    'Número partido doador': 'id_donator_party',
    'Sequencial Candidato': 'id_candidate_seq',
    'Sequencial Comite': 'id_commitee_seq',
    'Setor econômico do doador': 'cat_donator_economic_sector',
    'Setor econômico do doador originário': 'cat_original_donator_economic_sector',
    'Sigla  Partido': 'cat_party',
    'Sigla UE doador': 'cat_donator_state',
    'Tipo doador originário': 'cat_original_donator_type',
    'Tipo receita': 'cat_donation_type',
    'Tipo Comite': 'cat_commitee_type',
    'UF': 'cat_election_state',
    'Valor receita': 'num_donation_ammount',
}
# %%
print('Renaming Columns...')
comites_brasil.rename(mapper=translator, axis='columns', inplace=True)
partidos_brasil.rename(mapper=translator, axis='columns', inplace=True)
corruptos_brasil.rename(mapper=translator, axis='columns', inplace=True)
print('Done.')
# %% [markdown]
# Criamos uma coluna doador que, mapeia o doador intermediário no doador originário nos casos onde esse é nulo.
# O objetivo é que essa coluna possua a informação do doador original nos casos de repasse e o próprio doador nos casos de doações diretas.
print('Creating effective donator column..')
corruptos_brasil.loc[:, 'id_donator_effective_cpf_cnpj'] = corruptos_brasil.loc[:,
                                                                                'id_original_donator_cpf_cnpj']
corruptos_brasil['id_donator_effective_cpf_cnpj'].fillna(
    corruptos_brasil['id_donator_cpf_cnpj'], inplace=True)
print('Done.')

# %%
print('Formating donations values...')
comites_brasil.loc[:, 'num_donation_ammount'] = comites_brasil['num_donation_ammount'].str.replace(
    ',', '.').apply(float)
partidos_brasil.loc[:, 'num_donation_ammount'] = partidos_brasil['num_donation_ammount'].str.replace(
    ',', '.').apply(float)
corruptos_brasil.loc[:, 'num_donation_ammount'] = corruptos_brasil['num_donation_ammount'].str.replace(
    ',', '.').apply(float)
print('Done.')


# %%
print('Formating name values...')
corruptos_brasil.loc[:, "cat_donator_name"] = corruptos_brasil.cat_donator_name.apply(
    lambda x: unidecode.unidecode(x).replace('/', '').replace('.', '').strip() if type(x) == str else x)
corruptos_brasil.loc[:, "cat_donator_name2"] = corruptos_brasil.cat_donator_name2.apply(
    lambda x: unidecode.unidecode(x).replace('/', '').replace('.', '').strip() if type(x) == str else x)
corruptos_brasil.loc[:, "cat_original_donator_name"] = corruptos_brasil.cat_original_donator_name.apply(
    lambda x: unidecode.unidecode(x).replace('/', '').replace('.', '').strip() if type(x) == str else x)
corruptos_brasil.loc[:, "cat_original_donator_name2"] = corruptos_brasil.cat_original_donator_name2.apply(
    lambda x: unidecode.unidecode(x).replace('/', '').replace('.', '').strip() if type(x) == str else x)
corruptos_brasil.loc[:, "cat_donator_economic_sector"] = corruptos_brasil.cat_donator_economic_sector.apply(
    lambda x: unidecode.unidecode(x).replace('/', '').replace('.', '').strip() if type(x) == str else x)
corruptos_brasil.loc[:, "cat_original_donator_economic_sector"] = corruptos_brasil.cat_original_donator_economic_sector.apply(
    lambda x: unidecode.unidecode(x).replace('/', '').replace('.', '').strip() if type(x) == str else x)
print('Done.')
# %%
print('Creating english files...')
comites_brasil.to_csv(
    data_path / "receitas_comites_2014_brasil_english.csv", index=False)
partidos_brasil.to_csv(
    data_path / "receitas_partidos_2014_brasil_english.csv", index=False)
corruptos_brasil.to_csv(
    data_path / "receitas_candidatos_2014_brasil_english.csv", index=False)
print(len(corruptos_brasil))
print('Done.')


# %%
print('Creating candidates table...')
legenda_cantidatos = ['DATA_GERACAO', 'HORA_GERACAO', 'ANO_ELEICAO', 'NUM_TURNO',       'DESCRICAO_ELEICAO', 'SIGLA_UF', 'SIGLA_UE', 'DESCRICAO_UE',       'CODIGO_CARGO', 'DESCRICAO_CARGO', 'NOME_CANDIDATO',       'SEQUENCIAL_CANDIDATO', 'NUMERO_CANDIDATO', 'CPF_CANDIDATO',       'NOME_URNA_CANDIDATO', 'COD_SITUACAO_CANDIDATURA',       'DES_SITUACAO_CANDIDATURA', 'NUMERO_PARTIDO', 'SIGLA_PARTIDO',       'NOME_PARTIDO', 'CODIGO_LEGENDA', 'SIGLA_LEGENDA', 'COMPOSICAO_LEGENDA',       'NOME_LEGENDA', 'CODIGO_OCUPACAO',
                      'DESCRICAO_OCUPACAO',       'DATA_NASCIMENTO', 'NUM_TITULO_ELEITORAL_CANDIDATO',       'IDADE_DATA_ELEICAO', 'CODIGO_SEXO', 'DESCRICAO_SEXO',       'COD_GRAU_INSTRUCAO', 'DESCRICAO_GRAU_INSTRUCAO', 'CODIGO_ESTADO_CIVIL',       'DESCRICAO_ESTADO_CIVIL', 'CODIGO_COR_RACA', 'DESCRICAO_COR_RACA',       'CODIGO_NACIONALIDADE', 'DESCRICAO_NACIONALIDADE',       'SIGLA_UF_NASCIMENTO', 'CODIGO_MUNICIPIO_NASCIMENTO',       'NOME_MUNICIPIO_NASCIMENTO', 'DESPESA_MAX_CAMPANHA',       'COD_SIT_TOT_TURNO', 'DESC_SIT_TOT_TURNO', 'NM_EMAIL']
candidatos_brasil = pd.DataFrame()
path_2014 = raw_data_path / 'cadidatos/'
allFiles = glob.glob(os.path.join(path_2014, "consulta_cand_2014_*"))
candidatos_brasil = pd.concat((pd.read_csv(f, header=None, names=legenda_cantidatos,  encoding="ISO-8859-1",
                                           sep=';', na_values='#NULO#', dtype={'SEQUENCIAL_CANDIDATO': str}) for f in allFiles))
candidatos_brasil.to_csv(
    data_path / 'consulta_cand_2014_full.csv', index=False)


# Filtramos candidatos para os cargos estudos, cuja candidatura foi deferida

# %%
candidatos_deputado_estadual_indeferidos = candidatos_brasil[(
    candidatos_brasil['DES_SITUACAO_CANDIDATURA'] != "DEFERIDO") & ((candidatos_brasil['CODIGO_CARGO'] == 7))]
candidatos_deputado_estadual_indeferidos.to_csv(
    data_path / 'candidatos_deputado_estadual_indeferidos.csv', index=False)


# %%
candidatos_deputado_estadual_deferidos = candidatos_brasil[(
    candidatos_brasil['DES_SITUACAO_CANDIDATURA'] == "DEFERIDO") & ((candidatos_brasil['CODIGO_CARGO'] == 7))]
candidatos_deputado_estadual_deferidos.to_csv(
    data_path / 'candidatos_deputado_estadual_deferidos.csv', index=False)

print('Done.')

# %%
print('Creating votes table...')
legenda_votos = ['DATA_GERACAO', 'HORA_GERACAO', 'ANO_ELEICAO', 'NUM_TURNO', 'DESCRICAO_ELEICAO', 'SIGLA_UF', 'SIGLA_UE', 'CODIGO_MUNICIPIO', 'NOME_MUNICIPIO', 'NUMERO_ZONA', 'CODIGO_CARGO', 'NUMERO_CAND', 'SQ_CANDIDATO', 'NOME_CANDIDATO', 'NOME_URNA_CANDIDATO',       'DESCRICAO_CARGO',
                 'COD_SIT_CAND_SUPERIOR', 'DESC_SIT_CAND_SUPERIOR',       'CODIGO_SIT_CANDIDATO', 'DESC_SIT_CANDIDATO', 'CODIGO_SIT_CAND_TOT',       'DESC_SIT_CAND_TOT', 'NUMERO_PARTIDO', 'SIGLA_PARTIDO', 'NOME_PARTIDO',       'SEQUENCIAL_LEGENDA', 'NOME_COLIGACAO', 'COMPOSICAO_LEGENDA',       'TOTAL_VOTOS', 'TRANSITO']
votos_brasil = pd.DataFrame()
path_2014 = raw_data_path / 'votacao/'
allFiles = glob.glob(os.path.join(
    path_2014, "votacao_candidato_munzona_2014_*"))
votos_brasil = pd.concat((pd.read_csv(f, header=None, names=legenda_votos,  encoding="ISO-8859-1",
                                      sep=';', na_values='#NULO#', dtype={'SQ_CANDIDATO': str, 'CPF_CANDIDATO': str}) for f in allFiles))
votos_brasil.to_csv(
    data_path / 'votacao_candidato_munzona_2014_full.csv', index=False)


votos_deputados_estaduais_indeferidos = votos_brasil[(
    (votos_brasil["CODIGO_CARGO"] == 7)) & (votos_brasil["DESC_SIT_CANDIDATO"] != "DEFERIDO")]
votos_deputados_estaduais_indeferidos.to_csv(
    data_path / 'votos_deputados_estaduais_indeferidos.csv')

# %%
votos_deputados_estaduais_deferidos = votos_brasil[(
    (votos_brasil["CODIGO_CARGO"] == 7)) & (votos_brasil["DESC_SIT_CANDIDATO"] == "DEFERIDO")]
votos_deputados_estaduais_deferidos.to_csv(
    data_path / 'votos_deputados_estaduais_deferidos.csv')
print('Done.')


print('Merging Votes and Candidates by sequentials...')
# #### Vamos ralizar o merge indexado pelas sequencias, tomando a precaução de utlizar o método 'left' com a tabela de votos contendo o índice da esquerda, isso manterá a informação de Tâmara.

# %%
merged_votos_candidatos = pd.merge(votos_deputados_estaduais_deferidos,
                                   candidatos_deputado_estadual_deferidos,
                                   left_on='SQ_CANDIDATO',
                                   right_on="SEQUENCIAL_CANDIDATO",
                                   how='left', sort=False, suffixes=("_votes", "_candidatos"))

# Selecionamos colunas relevantes e eliminando duplicidades.

# %%
sub_set_cols = ['NUM_TURNO_votes',
                'SIGLA_UF_votes',
                'SIGLA_UE_votes',
                'CODIGO_MUNICIPIO',
                'NOME_MUNICIPIO',
                'NUMERO_ZONA',
                'CODIGO_CARGO_votes',
                'NUMERO_CAND',
                'SQ_CANDIDATO',
                'NOME_CANDIDATO_votes',
                'NUMERO_PARTIDO_votes',
                'SIGLA_PARTIDO_votes',
                'NOME_PARTIDO_votes',
                'SEQUENCIAL_LEGENDA',
                'NOME_COLIGACAO',
                'COMPOSICAO_LEGENDA_votes',
                'TOTAL_VOTOS',
                'SEQUENCIAL_CANDIDATO',
                'NUMERO_CANDIDATO',
                'CPF_CANDIDATO',
                'CODIGO_LEGENDA',
                'SIGLA_LEGENDA',
                'COMPOSICAO_LEGENDA_candidatos',
                "DESPESA_MAX_CAMPANHA",
                'CODIGO_SIT_CAND_TOT',
                'DESC_SIT_CAND_TOT']
sub_set_merged_votos_candidatos = merged_votos_candidatos.loc[:, sub_set_cols].copy(
)
sub_set_merged_votos_candidatos.to_csv(
    data_path / 'votos_candidatos__deputado_estadual_deferidos.csv')
print('Done.')


print('Aggregating candidate and votes infos...')
# %% [markdown]
# Vamos criar uma tabela contendo informações do canditato, com o número de votos por munícipio somados.

# %%
votacao_candidato_deputado_estadual = sub_set_merged_votos_candidatos.groupby(
    ["CPF_CANDIDATO", 'DESC_SIT_CAND_TOT', 'SIGLA_UF_votes', 'NOME_CANDIDATO_votes', 'COMPOSICAO_LEGENDA_votes', 'SIGLA_PARTIDO_votes']).agg({'TOTAL_VOTOS': 'sum'}).reset_index()
votacao_candidato_deputado_estadual.to_csv(
    data_path / "votacao_candidato_deputado_estadual_deferidos_soma_municipios.csv")

print('Done.')

print('Getting info about dirty CPF/CNPJ...')


# %% [markdown]
# ### Comparing with CNEP and CEIS data.
# - http://www.portaltransparencia.gov.br/download-de-dados/cnep
# - http://www.portaltransparencia.gov.br/download-de-dados/ceis

# %%
CNEP = pd.read_csv('http://www.portaltransparencia.gov.br/download-de-dados/cnep/20190315',
                   compression='zip', sep=';', encoding='iso-8859-1')
CEIS = pd.read_csv('http://www.portaltransparencia.gov.br/download-de-dados/ceis/20190315',
                   compression='zip', sep=';', encoding='iso-8859-1')
CNEP.to_csv(data_path / '20190301_CNEP.csv', index=False)
CEIS.to_csv(data_path / '20190301_CEIS.csv', index=False)

print('Done.')
