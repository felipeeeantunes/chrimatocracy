from pathlib import Path

import pandas as pd


def load_donations(role: str, year: int, data_path: Path):

    data_types = {
        "id_accountant_cnpj": str,
        "id_candidate_cpf": str,
        "id_donator_cpf_cnpj": str,
        "id_original_donator_cpf_cnpj": str,
        "id_donator_effective_cpf_cnpj": str,
    }

    candidatos = pd.read_csv(
        data_path / f"brazil_{year}_{role}_candidates_donations.csv",
        low_memory=False,
        dtype=data_types,
    )

    df = candidatos[(candidatos["id_donator_effective_cpf_cnpj"].isnull() == False)]

    return df


def donations_received_stats(df: pd.DataFrame, year: int, role: str, table_path: Path):

    stats_donations = pd.DataFrame(df["num_donation_ammount"].describe()).T.append(
        pd.DataFrame(
            df[df["id_donator_effective_cpf_cnpj"].astype(str).apply(lambda x: len(x)) == 14][
                "num_donation_ammount"
            ].describe()
        ).T.append(
            pd.DataFrame(
                df[df["id_donator_effective_cpf_cnpj"].astype(str).apply(lambda x: len(x)) == 11][
                    "num_donation_ammount"
                ].describe()
            ).T
        )
    )
    stats_donations.index = ["All", "CNPJ", "CPF"]
    stats_donations.columns = ["N", "Avg", "Std", "Min", "25%", "50%", "75%", "Max"]

    with open(table_path / f"brazil_{year}_{role}_donations_to_candidates_stats.tex", "w") as tf:
        tf.write(
            stats_donations.to_latex(
                escape=True,
                bold_rows=True,
                formatters={
                    "N": "{:.0f}".format,
                    "Avg (BRL)": "{:.2f}".format,
                    "Std (BRL)": "{:.2f}".format,
                    "Min (BRL)": "{:.2f}".format,
                    "25% (BRL)": "{:.2f}".format,
                    "50% (BRL)": "{:.2f}".format,
                    "75% (BRL)": "{:.2f}".format,
                    "Max (BRL)": "{:.2f}".format,
                },
            )
        )
    return stats_donations


def donations_made_stats(df: pd.DataFrame, by: str, year: int, role: str, table_path: Path):
    col_map = dict(
        companies="cat_original_donator_name2",
        sector="cat_original_donator_economic_sector",
    )
    g_df = df.groupby(col_map.get(by)).agg(
        {
            "id_candidate_cpf": lambda x: x.nunique(),
            "id_election": lambda x: len(x),
            "num_donation_ammount": lambda x: [x.sum(), x.mean(), x.std()],
        }
    )
    table = g_df.sort_values(by="num_donation_ammount", ascending=False).reset_index()

    table["Total (BRL)"] = table["num_donation_ammount"].apply(lambda x: x[0])
    table["Mean (BRL)"] = table["num_donation_ammount"].apply(lambda x: x[1])
    table["Std (BRL) "] = table["num_donation_ammount"].apply(lambda x: x[2])
    table.drop("num_donation_ammount", axis=1, inplace=True)

    petry_name_map = dict(
        companies="Company Name",
        sector="Economic Sector",
    )
    table.columns = [
        petry_name_map.get(by),
        "Number of Candidates",
        "Number of Donations",
        "Total (BRL)",
        "Mean (BRL)",
        "Standard Deviation (BRL)",
    ]

    n = len(table)
    n_candidates = table["Number of Candidates"].sum()
    n_donations = table["Number of Donations"].sum()
    total_donations = table["Total (BRL)"].sum()

    print(
        f"""
        The 3 {petry_name_map.get(by)} that most donated to candidates are: {[i for i in table[petry_name_map.get(by)][:3].values]}.\n
        Alone, they are reponsable for {round(table["Number of Donations"][:3].sum() / n_donations * 100, 1)} % of all donations.\n
        Corresponding to {round(table["Total (BRL)"][:3].sum() / total_donations * 100, 1)} % the total ammount donated.\n
        These 3 {petry_name_map.get(by)} correspond to {round(3 / n * 100, 3)} % the companies who donated.
        """
    )

    with open(table_path / f"brazil_{year}_{role}_donations_made_by_{by}_stats.tex", "w") as tf:
        tf.write(table[table["Number of Candidates"] > 0.01 * n_candidates].to_latex(index=False))

    return table
