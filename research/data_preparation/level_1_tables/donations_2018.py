# %%
import glob
import os
import sys
from pathlib import Path

import pandas as pd
import unidecode

parent_dir = Path().cwd()
# %%
raw_data_path = Path(parent_dir) / "raw_data/2018/"
data_path = Path(parent_dir) / "data/2018/"

if not os.path.exists(data_path):
    os.makedirs(data_path)

receitas_candidatos_brasil = pd.read_csv(
    raw_data_path / "receitas_candidatos_2018_BRASIL.csv",
    encoding="ISO-8859-1",
    sep=";",
    na_values=[-1, "#NULO#"],
    dtype=str,
    decimal=",",
)

receitas_doador_originario = pd.read_csv(
    raw_data_path / "receitas_candidatos_2018_BRASIL.csv",
    encoding="ISO-8859-1",
    sep=";",
    na_values=[-1, "#NULO#"],
    dtype=str,
    decimal=",",
)

translator = {
    "NR_CNPJ_PRESTADOR_CONTA": "id_accountant_cnpj",
    "NR_CPF_CANDIDATO": "id_candidate_cpf",
    "NR_CPF_CNPJ_DOADOR": "id_donator_cpf_cnpj",
    #'NR_CPF_CNPJ_DOADOR_ORIGINARIO': 'id_original_donator_cpf_cnpj',
    "DS_CARGO": "cat_role_description",
    "CD_CNAE_DOADOR": "id_donator_economic_sector",
    "CD_TIPO_ELEICAO": "id_election",
    "DT_RECEITA": "dat_donation_date",
    "HH_RECEITA": "dat_donation_time",
    "DS_ELEICAO": "cat_election_description",
    "DS_FONTE_RECEITA": "cat_donation_description",
    "DS_ESPECIE_RECEITA": "cat_donation_method",
    "DS_ORIGEM_RECEITA": "cat_donation_source",
    "NM_CANDIDATO": "cat_candidate_name",
    "NM_DOADOR": "cat_donator_name",
    "NM_DOADOR_RFB": "cat_donator_name2",
    "NM_DOADOR_ORIGINARIO": "cat_original_donator_name",
    "NM_DOADOR_ORIGINARIO_RFB": "cat_original_donator_name2",
    "NR_RECIBO_DOACAO": "id_receipt_num",
    "NM_CANDIDATO": "id_candidate_num",
    "NR_DOCUMENTO_DOACAO": "id_document_num",
    "NR_CANDIDATO_DOADOR": "id_donator_number",
    "NR_PARTIDO_DOADOR": "id_donator_party",
    "SQ_PRESTADOR_CONTAS": "id_candidate_seq",
    #'Sequencial Comite': 'id_commitee_seq',
    "DS_CNAE_DOADOR": "cat_donator_economic_sector",
    #'Setor econômico do doador originário': 'cat_original_donator_economic_sector',
    "SG_PARTIDO": "cat_party",
    "SG_UF_DOADOR": "cat_donator_state",
    "TP_DOADOR_ORIGINARIO": "cat_original_donator_type",
    "DS_NATUREZA_RECEITA": "cat_donation_type",
    # 'Tipo Comite': 'cat_commitee_type',
    "SG_UE": "cat_federative_unity",
    "VR_RECEITA": "num_donation_ammount",
}

print("Renaming Columns...")
receitas_candidatos_brasil.rename(mapper=translator, axis="columns", inplace=True)
print("Done.")

print("Creating effective donator column..")
receitas_candidatos_brasil["id_donator_effective_cpf_cnpj"] = None
receitas_candidatos_brasil["id_donator_effective_cpf_cnpj"].fillna(
    receitas_candidatos_brasil["id_donator_cpf_cnpj"], inplace=True
)
print("Done.")

# %%
print("Formating donations values...")
receitas_candidatos_brasil.loc[:, "num_donation_ammount"] = (
    receitas_candidatos_brasil["num_donation_ammount"].str.replace(",", ".").apply(float)
)
print("Done.")

print("Formating role names...")
receitas_candidatos_brasil.loc[:, "cat_role_description"] = receitas_candidatos_brasil[
    "cat_role_description"
].str.upper()
print("Done.")


# %%
print("Formating name values...")
receitas_candidatos_brasil.loc[:, "cat_donator_name"] = receitas_candidatos_brasil.cat_donator_name.apply(
    lambda x: unidecode.unidecode(x).replace("/", "").replace(".", "").strip() if type(x) == str else x
)
receitas_candidatos_brasil.loc[:, "cat_donator_name2"] = receitas_candidatos_brasil.cat_donator_name2.apply(
    lambda x: unidecode.unidecode(x).replace("/", "").replace(".", "").strip() if type(x) == str else x
)
# receitas_candidatos_brasil.loc[:, "cat_original_donator_name"] = receitas_candidatos_brasil.cat_original_donator_name.apply(
#     lambda x: unidecode.unidecode(x).replace('/', '').replace('.', '').strip() if type(x) == str else x)
# receitas_candidatos_brasil.loc[:, "cat_original_donator_name2"] = receitas_candidatos_brasil.cat_original_donator_name2.apply(
#     lambda x: unidecode.unidecode(x).replace('/', '').replace('.', '').strip() if type(x) == str else x)
receitas_candidatos_brasil.loc[
    :, "cat_donator_economic_sector"
] = receitas_candidatos_brasil.cat_donator_economic_sector.apply(
    lambda x: unidecode.unidecode(x).replace("/", "").replace(".", "").strip() if type(x) == str else x
)
# receitas_candidatos_brasil.loc[:, "cat_original_donator_economic_sector"] = receitas_candidatos_brasil.cat_original_donator_economic_sector.apply(
#    lambda x: unidecode.unidecode(x).replace('/', '').replace('.', '').strip() if type(x) == str else x)
print("Done.")
# %%
print("Creating english files...")
receitas_candidatos_brasil.to_csv(data_path / "brazil_2018_donations_candidates.csv", index=False)
print("Done.")
