import os
import sys
from pathlib import Path

import pandas as pd

parent_dir = Path().cwd()
# %%
raw_data_path = Path(parent_dir) / "raw_data/2014/"
data_path = Path(parent_dir) / "data/"

if not os.path.exists(data_path):
    os.makedirs(data_path)

data_types = {
    "id_accountant_cnpj": str,
    "id_candidate_cpf": str,
    "id_donator_cpf_cnpj": str,
    "id_original_donator_cpf_cnpj": str,
    "id_donator_effective_cpf_cnpj": str,
}

votos_brasil = pd.read_csv(data_path / "brazil_2014_votes.csv", low_memory=False, dtype=data_types)

role_pairs = votos_brasil[["cat_role_description", "id_role_code"]].drop_duplicates().values
translator = {
    "GOVERNADOR": "governor",
    "SENADOR": "senator",
    "DEPUTADO FEDERAL": "federal_deputy",
    "DEPUTADO ESTADUAL": "state_deputy",
    "DEPUTADO DISTRITAL": "district_deputy",
    "PRESIDENTE": "president",
    "DEFERIDO": "accepted",
    "INDEFERIDO": "non_accepted",
}

for role, code in role_pairs:
    for situation in ["DEFERIDO", "INDEFERIDO"]:
        print(f"Generating votes table for {role} with {situation} situation.\n")
        votos = votos_brasil[
            ((votos_brasil["id_role_code"] == code)) & (votos_brasil["cat_sit_candidate"] == situation)
        ]
        translated_role = translator.get(role, role.lower().strip())
        translated_situation = translator.get(situation)

        votos.to_csv(
            data_path / f"brazil_2014_{translated_role}_{translated_situation}_candidates_votes.csv", index=False
        )

        print("Done.")
