import os
import sys
from pathlib import Path

import pandas as pd

parent_dir = Path().cwd()
# %%
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

candidatos_brasil = pd.read_csv(data_path / "brazil_2014_candidates.csv", low_memory=False, dtype=data_types)

role_pairs = candidatos_brasil[["cat_role_description", "id_role_code"]].drop_duplicates().values
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
        print(f"Generating candidates table for {role} with {situation} situation.\n")
        candidatos = candidatos_brasil[
            ((candidatos_brasil["id_role_code"] == code))
            & (candidatos_brasil["cat_sit_candidate_description"] == situation)
        ]
        translated_role = translator.get(role, role.lower().strip())
        translated_situation = translator.get(situation)

        candidatos.to_csv(
            data_path / f"brazil_2014_{translated_role}_{translated_situation}_candidates.csv", index=False
        )

        print("Done.")
