n_candidatos = candidatos["id_candidate_cpf"].nunique()
n_doacoes = candidatos.shape[0]
tot_doacoes = candidatos["num_donation_ammount"].sum()
g_canditato = candidatos.groupby("id_candidate_cpf").agg(
    {"id_election": lambda x: len(x), "num_donation_ammount": lambda x: x.sum()}
)
g_especie = candidatos.groupby("cat_donation_method").agg(
    {"id_candidate_cpf": lambda x: x.nunique(), "id_election": lambda x: len(x), "num_donation_ammount": sum}
)
g_setor = candidatos.groupby("cat_original_donator_economic_sector").agg(
    {
        "id_candidate_cpf": lambda x: x.nunique(),
        "id_election": lambda x: len(x),
        "num_donation_ammount": lambda x: [x.sum(), x.mean(), x.std()],
    }
)
g_uf = candidatos.groupby("cat_federative_unity").agg(
    {"id_candidate_cpf": lambda x: x.nunique(), "id_election": lambda x: len(x), "num_donation_ammount": sum}
)
g_partido = candidatos.groupby("cat_party").agg(
    {"id_candidate_cpf": lambda x: x.nunique(), "id_election": lambda x: len(x), "num_donation_ammount": sum}
)
g_setor_uf = candidatos.groupby(["cat_original_donator_economic_sector", "cat_federative_unity"]).agg(
    {
        "id_candidate_cpf": lambda x: x.nunique(),
        "id_election": lambda x: len(x),
        "num_donation_ammount": lambda x: [x.sum(), x.mean(), x.std()],
    }
)
g_setor_partido = candidatos.groupby(["cat_party", "cat_original_donator_economic_sector"]).agg(
    {"id_candidate_cpf": lambda x: x.nunique(), "id_election": lambda x: len(x), "num_donation_ammount": sum}
)


# The donations cover unniformily seven orders of magnitude

# ## How different economic sectors donate?


g_sector = candidatos.groupby("cat_original_donator_economic_sector").agg(
    {
        "id_candidate_cpf": lambda x: x.nunique(),
        "id_election": lambda x: len(x),
        "num_donation_ammount": lambda x: [x.sum(), x.mean(), x.std()],
    }
)
tab_sector = g_sector.sort_values(by="num_donation_ammount", ascending=False).reset_index()


tab_sector


tab_sector["Total Ammont"] = tab_sector["num_donation_ammount"].apply(lambda x: x[0])
tab_sector["Mean Ammount"] = tab_sector["num_donation_ammount"].apply(lambda x: x[1])
tab_sector["Std  Ammount "] = tab_sector["num_donation_ammount"].apply(lambda x: x[2])
tab_sector.drop("num_donation_ammount", axis=1, inplace=True)


tab_sector.columns = [
    "Economic Sector",
    "Number of Candidates",
    "Number of Donations",
    "Total (R$)",
    "Mean (R$)",
    "Standard Deviation (R$)",
]


n_donations = tab_sector["Number of Donations"].sum()
total_donations = tab_sector["Total (R$)"].sum()


tab_sector[tab_sector["Number of Candidates"] > 0.01 * n_candidatos]


print("Donation Statistics by Economic Sector:\n", tab_sector[tab_sector["Number of Candidates"] > 0.01 * n_candidatos])
with open(table_path / f"brazil_{self.year}_{role}_statistics_by_sector.tex", "w") as tf:
    tf.write(tab_sector[tab_sector["Number of Candidates"] > 0.01 * n_candidatos].to_latex(index=False))


print("The 3 economic sectors that most donated to candidates are:")
[print(i) for i in tab_sector["Economic Sector"][:3].values]


print(
    "Alone, they are reponsable from",
    round(tab_sector["Number of Donations"][:3].sum() / n_donations * 100, 1),
    "% of all donations.",
)


print(
    "Corresponding to",
    round(tab_sector["Total (R$)"][:3].sum() / total_donations * 100, 1),
    "% the total ammount donated.",
)


# ## How much the companies donated?


g_companies = candidatos.groupby("cat_original_donator_name2").agg(
    {
        "id_candidate_cpf": lambda x: x.nunique(),
        "id_election": lambda x: len(x),
        "num_donation_ammount": lambda x: [x.sum(), x.mean(), x.std()],
    }
)
tab_companies = g_companies.sort_values(by="num_donation_ammount", ascending=False).reset_index()


tab_companies["Total Ammont"] = tab_companies["num_donation_ammount"].apply(lambda x: x[0])
tab_companies["Mean Ammount"] = tab_companies["num_donation_ammount"].apply(lambda x: x[1])
tab_companies["Std  Ammount "] = tab_companies["num_donation_ammount"].apply(lambda x: x[2])
tab_companies.drop("num_donation_ammount", axis=1, inplace=True)


tab_companies.columns = [
    "Company Name",
    "Number of Candidates",
    "Number of Donations",
    "Total (R$)",
    "Mean (R$)",
    "Standard Deviation (R$)",
]


print("Donation Statistics by Company\n", tab_companies[tab_companies["Number of Candidates"] > 0.01 * n_candidatos])
with open(table_path / f"brazil_{self.year}_{role}_statistics_by_company.tex", "w") as tf:
    tf.write(tab_companies[tab_companies["Number of Candidates"] > 0.01 * n_candidatos].to_latex(index=False))


print("The 3 companies that most donated to candidates are:")
[print(i) for i in tab_companies["Company Name"][:3].values]


print(
    "Alone, they are reponsable for",
    round(tab_companies["Number of Donations"][:3].sum() / n_donations * 100, 1),
    "% of all donations.",
)


print(
    "Corresponding to",
    round(tab_companies["Total (R$)"][:3].sum() / total_donations * 100, 1),
    "% the total ammount donated.",
)


n_companies = len(tab_companies)


print("This 3 companies correspond to", round(3 / n_companies * 100, 3), "% the companies who donated.")


top_3_companies = ["JBS S/A", "CONSTRUTORA QUEIROZ GALVAO S A", "U T C ENGENHARIA S/A"]


donator_top3_companies = candidatos[candidatos["cat_original_donator_name2"].isin(top_3_companies)]
g_parties = donator_top3_companies.groupby("cat_party").agg(
    {
        "cat_original_donator_name2": lambda x: x.nunique(),
        "id_election": lambda x: len(x),
        "num_donation_ammount": lambda x: [x.sum(), x.mean(), x.std()],
    }
)
tab_parties = g_parties.sort_values(by="num_donation_ammount", ascending=False).reset_index()


tab_parties["Total Ammont"] = tab_parties["num_donation_ammount"].apply(lambda x: x[0])
tab_parties["Mean Ammount"] = tab_parties["num_donation_ammount"].apply(lambda x: x[1])
tab_parties["Std  Ammount "] = tab_parties["num_donation_ammount"].apply(lambda x: x[2])
tab_parties.drop("num_donation_ammount", axis=1, inplace=True)
tab_parties.reset_index()


tab_parties.columns = [
    "Party",
    "Number of Companies",
    "Number of Donations",
    "Total (R$)",
    "Mean (R$)",
    "Standard Deviation (R$)",
]


print("Donation Statistics by Party\n", tab_parties)
with open(table_path / f"brazil_{self.year}_{role}_statistics_by_party.tex", "w") as tf:
    tf.write(tab_parties.to_latex(index=False))


print("The 3 parties most benefited from top 3 companies:")
[print(i) for i in tab_parties["Party"][:3].values]


n_donations = tab_parties["Number of Donations"].sum()
total_donations = tab_parties["Total (R$)"].sum()


print(
    "Alone, they are reponsable for",
    round(tab_parties["Number of Donations"][:3].sum() / n_donations * 100, 2),
    "% of all donations.",
)


print(
    "Corresponding to",
    round(tab_parties["Total (R$)"][:3].sum() / total_donations * 100, 2),
    "% the total ammount donated.",
)


# ## How many donation each candidate received?


g_candidates = candidatos.groupby("id_candidate_cpf").agg(
    {
        "cat_original_donator_name2": lambda x: x.nunique(),
        "id_election": lambda x: len(x),
        "num_donation_ammount": lambda x: [x.sum(), x.mean(), x.std()],
    }
)
tab_candidates = g_candidates.sort_values(by="num_donation_ammount", ascending=False).reset_index()


tab_candidates["Total Ammont"] = tab_candidates["num_donation_ammount"].apply(lambda x: x[0])
tab_candidates["Mean Ammount"] = tab_candidates["num_donation_ammount"].apply(lambda x: x[1])
tab_candidates["Std  Ammount "] = tab_candidates["num_donation_ammount"].apply(lambda x: x[2])
tab_candidates.drop("num_donation_ammount", axis=1, inplace=True)


tab_candidates.columns = [
    "Candidate CPF",
    "Number of Companies",
    "Number of Donations",
    "Total (R$)",
    "Mean (R$)",
    "Standard Deviation (R$)",
]

print("Donation Statistics by Candidate\n", tab_candidates)
with open(table_path / f"brazil_{self.year}_{role}_statistics_by_candidate.tex", "w") as tf:
    tf.write(tab_candidates.to_latex(index=False))
