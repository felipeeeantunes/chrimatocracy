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
print("Reading and decoding files...")
receitas_comites_brasil = pd.read_csv(
    raw_data_path / "receitas_comites_2014_brasil.txt", encoding="ISO-8859-1", sep=";", na_values="#NULO", dtype=str
)
receitas_partidos_brasil = pd.read_csv(
    raw_data_path / "receitas_partidos_2014_brasil.txt", encoding="ISO-8859-1", sep=";", na_values="#NULO", dtype=str
)
receitas_candidatos_brasil = pd.read_csv(
    raw_data_path / "receitas_candidatos_2014_brasil.txt", encoding="ISO-8859-1", sep=";", na_values="#NULO", dtype=str
)
print("Done.")

# %% [markdown]
# ## Cria dicionario com colunnas em inglês
# %%
translator = {
    "CNPJ Prestador Conta": "id_accountant_cnpj",
    "CPF do candidato": "id_candidate_cpf",
    "CPF/CNPJ do doador": "id_donator_cpf_cnpj",
    "CPF/CNPJ do doador originário": "id_original_donator_cpf_cnpj",
    "Cargo": "cat_role_description",
    "Cod setor econômico do doador": "id_donator_economic_sector",
    "Cód. Eleição": "id_election",
    "Data da receita": "dat_donation_date",
    "Data e hora": "dat_donation_date_time",
    "Desc. Eleição": "cat_election_description",
    "Descricao da receita": "cat_donation_description",
    "Especie recurso": "cat_donation_method",
    "Fonte recurso": "cat_donation_source",
    "Nome candidato": "cat_candidate_name",
    "Nome do doador": "cat_donator_name",
    "Nome do doador (Receita Federal)": "cat_donator_name2",
    "Nome do doador originário": "cat_original_donator_name",
    "Nome do doador originário (Receita Federal)": "cat_original_donator_name2",
    "Numero Recibo Eleitoral": "id_receipt_num",
    "Numero candidato": "id_candidate_num",
    "Numero do documento": "id_document_num",
    "Número candidato doador": "id_donator_number",
    "Número partido doador": "id_donator_party",
    "Sequencial Candidato": "id_candidate_seq",
    "Sequencial Comite": "id_commitee_seq",
    "Setor econômico do doador": "cat_donator_economic_sector",
    "Setor econômico do doador originário": "cat_original_donator_economic_sector",
    "Sigla  Partido": "cat_party",
    "Sigla UE doador": "cat_donator_state",
    "Tipo doador originário": "cat_original_donator_type",
    "Tipo receita": "cat_donation_type",
    "Tipo Comite": "cat_commitee_type",
    "UF": "cat_federative_unity",
    "Valor receita": "num_donation_ammount",
}
# %%
print("Renaming Columns...")
receitas_comites_brasil.rename(mapper=translator, axis="columns", inplace=True)
receitas_partidos_brasil.rename(mapper=translator, axis="columns", inplace=True)
receitas_candidatos_brasil.rename(mapper=translator, axis="columns", inplace=True)
print("Done.")
# %% [markdown]
# Criamos uma coluna doador que, mapeia o doador intermediário no doador originário nos casos onde esse é nulo.
# O objetivo é que essa coluna possua a informação do doador original nos casos de repasse e o próprio doador nos casos de doações diretas.
print("Creating effective donator column..")
receitas_candidatos_brasil.loc[:, "id_donator_effective_cpf_cnpj"] = receitas_candidatos_brasil.loc[
    :, "id_original_donator_cpf_cnpj"
]
receitas_candidatos_brasil["id_donator_effective_cpf_cnpj"].fillna(
    receitas_candidatos_brasil["id_donator_cpf_cnpj"], inplace=True
)
print("Done.")

# %%
print("Formating donations values...")
receitas_comites_brasil.loc[:, "num_donation_ammount"] = (
    receitas_comites_brasil["num_donation_ammount"].str.replace(",", ".").apply(float)
)
receitas_partidos_brasil.loc[:, "num_donation_ammount"] = (
    receitas_partidos_brasil["num_donation_ammount"].str.replace(",", ".").apply(float)
)
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
receitas_candidatos_brasil.loc[
    :, "cat_original_donator_name"
] = receitas_candidatos_brasil.cat_original_donator_name.apply(
    lambda x: unidecode.unidecode(x).replace("/", "").replace(".", "").strip() if type(x) == str else x
)
receitas_candidatos_brasil.loc[
    :, "cat_original_donator_name2"
] = receitas_candidatos_brasil.cat_original_donator_name2.apply(
    lambda x: unidecode.unidecode(x).replace("/", "").replace(".", "").strip() if type(x) == str else x
)
receitas_candidatos_brasil.loc[
    :, "cat_donator_economic_sector"
] = receitas_candidatos_brasil.cat_donator_economic_sector.apply(
    lambda x: unidecode.unidecode(x).replace("/", "").replace(".", "").strip() if type(x) == str else x
)
receitas_candidatos_brasil.loc[
    :, "cat_original_donator_economic_sector"
] = receitas_candidatos_brasil.cat_original_donator_economic_sector.apply(
    lambda x: unidecode.unidecode(x).replace("/", "").replace(".", "").strip() if type(x) == str else x
)
print("Done.")
# %%
print("Creating english files...")
receitas_comites_brasil.to_csv(data_path / "brazil_2014_donations_committees.csv", index=False)
receitas_partidos_brasil.to_csv(data_path / "brazil_2014_donations_parties.csv", index=False)
receitas_candidatos_brasil.to_csv(data_path / "brazil_2014_donations_candidates.csv", index=False)
print("Done.")
