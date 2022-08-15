import os
import sys
from pathlib import Path

import pandas as pd

parent_dir = Path().cwd()
# %%
year = 2018
raw_data_path = Path(parent_dir) / f"raw_data/{year}/"
data_path = Path(parent_dir) / f"data/{year}"

if not os.path.exists(data_path):
    os.makedirs(data_path)

data_types = {
    "id_accountant_cnpj": str,
    "id_candidate_cpf": str,
    "id_donator_cpf_cnpj": str,
    #'id_original_donator_cpf_cnpj':str,
    "id_donator_effective_cpf_cnpj": str,
}

receitas_candidatos_brasil = pd.read_csv(
    data_path / f"brazil_{year}_donations_candidates.csv", low_memory=False, dtype=data_types
)


roles = [x.upper() for x in receitas_candidatos_brasil["cat_role_description"].unique()]
translator = {
    "GOVERNADOR": "governor",
    "SENADOR": "senator",
    "DEPUTADO FEDERAL": "federal_deputy",
    "DEPUTADO ESTADUAL": "state_deputy",
    "DEPUTADO DISTRITAL": "district_deputy",
    "PRESIDENTE": "president",
}

for role in roles:
    print(f"Generating donations table for {role}.\n")

    receitas = receitas_candidatos_brasil[(receitas_candidatos_brasil["cat_role_description"] == role)]
    translated_role = translator.get(role, role.lower().strip())

    receitas.to_csv(data_path / f"brazil_{year}_{translated_role}_candidates_donations.csv", index=False)
    print("Done.")

    print(f"Aggregating donations table for {role}.\n")
    g_candidates = receitas.groupby(
        [
            "id_candidate_cpf",
            "id_donator_effective_cpf_cnpj",
            "cat_party",
            "cat_role_description",
            "cat_federative_unity",
        ]
    ).agg({"num_donation_ammount": "sum"})
    g_candidates = pd.DataFrame(g_candidates)
    g_candidates = g_candidates.reset_index()
    g_candidates.to_csv(data_path / f"brazil_{year}_{translated_role}_candidates_donations_aggregated.csv", index=False)

    print("Done.")
