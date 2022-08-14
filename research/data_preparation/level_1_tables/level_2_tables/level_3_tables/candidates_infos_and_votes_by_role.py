# %%
import glob
import os
import sys
from pathlib import Path

import pandas as pd
import unidecode

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


roles = ["governor", "senator", "federal_deputy", "state_deputy", "district_deputy", "president"]
situations = ["accepted", "non_accepted"]


for role in roles:
    for situation in situations:
        print(f"Generating votes table for {role} with {situation} situation.\n")

        candidatos = pd.read_csv(
            data_path / f"brazil_2014_{role}_{situation}_candidates.csv", dtype=data_types, low_memory=False
        )
        votos = pd.read_csv(
            data_path / f"brazil_2014_{role}_{situation}_candidates_votes.csv", dtype=data_types, low_memory=False
        )

        print("Merging Votes and Candidates by sequentials...")
        # #### Vamos ralizar o merge indexado pelas sequencias, tomando a precaução de utlizar o método 'left' com a tabela de votos contendo o índice da esquerda, isso manterá a informação de Tâmara.

        # %%
        merge_index = [
            "id_candidate_sequential",
            "id_candidate_number",
            "int_election_turn",
            "cat_federative_unity",
            "cat_state_unity",
            "id_role_code",
        ]
        grouped_votos_candidates_municipios = votos.groupby(merge_index).agg({"int_number_of_votes": sum})
        try:
            merged_votos_candidatos = pd.merge(
                candidatos.set_index(merge_index),
                grouped_votos_candidates_municipios,
                left_index=True,
                right_index=True,
                how="left",
                sort=False,
            ).reset_index()

            # %%
            merged_votos_candidatos.to_csv(
                data_path / f"brazil_2014_{role}_{situation}_candidates_votes_with_candidates_info.csv", index=False
            )
        except:
            print(f"Can't create this file for {role} {situation}")

        print("Done.")
