import logging

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.api as sm
from matplotlib import ticker
from scipy import stats
from statsmodels.discrete.discrete_model import Logit

mpl.use("pgf")

pgf_with_pdflatex = {
    "pgf.texsystem": "pdflatex",
    "pgf.preamble": "\n".join(
        [
            r"\usepackage[utf8x]{inputenc}",
            r"\usepackage[T1]{fontenc}",
            r"\usepackage{cmbright}",
        ]
    ),
}
mpl.rcParams.update(pgf_with_pdflatex)
mpl.rcParams["figure.figsize"] = (4.7747, 3.5)


from chrimatocracy.model import GenerativeModel
from chrimatocracy.utils import benford as bf


class Donations(GenerativeModel):
    def __init__(
        self,
        role,
        year,
        data_path,
        table_path,
        figure_path,
        group,
        should_fit=True,
        benford_log_scale=True,
        logger: logging.Logger = None,
    ):

        super().__init__(
            year,
            role,
            data_path,
            table_path,
            figure_path,
            group,
            logger,
        )

        self.data_types = {
            "id_accountant_cnpj": str,
            "id_candidate_cpf": str,
            "id_donator_cpf_cnpj": str,
            "id_original_donator_cpf_cnpj": str,
            "id_donator_effective_cpf_cnpj": str,
        }

        self.should_fit = should_fit
        self.benford_log_scale = benford_log_scale
        self.logger = logger

    def load_donations(self):
        self.receitas = pd.read_csv(
            self.data_path / f"brazil_{self.year}_{self.role}_candidates_donations.csv",
            dtype=self.data_types,
            low_memory=False,
        )
        self.merged_receitas = pd.read_csv(
            self.data_path
            / f"brazil_{self.year}_{self.role}_accepted_candidates_votes_with_candidate_info_and_donations.csv",
            dtype=self.data_types,
            low_memory=False,
        )

    def benford_plot(self):
        self.all_digits = bf.read_numbers(self.receitas["num_donation_ammount"])
        self.all_probs = bf.find_probabilities(self.all_digits)

        x2, _ = bf.find_x2(pd.Series(self.all_digits))

        width = 0.2

        indx = np.arange(1, len(self.all_probs) + 1)
        benford = [np.log10(1 + (1.0 / d)) for d in indx]

        plt.bar(indx, benford, width, color="r", label="Lei de Benford", alpha=0.2)
        plt.bar(
            indx + width,
            self.all_probs,
            width,
            color="#1DACD6",
            label=r"Doações($\chi^2$" "=" + str(round(x2, 2)) + ")",
            alpha=0.2,
        )

        if self.benford_log_scale:
            plt.yscale("log")
            plt.xscale("log")
        ax = plt.gca()
        ax.set_xticks(indx)
        ax.set_yticks([0.05, 0.1, 0.2, 0.3])
        ax.get_xaxis().set_major_formatter(ticker.ScalarFormatter())
        ax.get_yaxis().set_major_formatter(ticker.ScalarFormatter())
        plt.title("Lei de Benford")
        plt.ylabel("Probabilidade")
        plt.grid(False)
        plt.legend()
        plt.tight_layout()

        plt.savefig(self.figure_path / f"brazil_{self.year}_{self.role}_benford_distribution.pgf")
        # plt.show()

    def lr_table(self):

        g_key = ["id_candidate_cpf", "cat_federative_unity", "int_election_turn", "cat_sit_tot_turn_description"]
        g_merged_receitas = (
            self.merged_receitas.groupby(g_key)
            .agg({"int_number_of_votes": lambda x: sum(x) / len(x), "num_donation_ammount": "sum"})
            .reset_index()
        )

        r_key = ["id_candidate_cpf", "cat_federative_unity", "int_election_turn"]
        rg_key = ["id_candidate_cpf", "cat_federative_unity"]

        raw_data = (
            g_merged_receitas.sort_values(by=r_key)
            .groupby(rg_key)
            .agg({"cat_sit_tot_turn_description": lambda x: np.array(x)[-1], "num_donation_ammount": "sum"})
            .reset_index()
        )

        x1 = raw_data["num_donation_ammount"]
        y = raw_data["cat_sit_tot_turn_description"].apply(
            lambda x: 1 if x in ["ELEITO POR QP", "ELEITO POR MÉDIA", "ELEITO"] else 0
        )

        soma = x1.sum()

        x1 = x1.apply(lambda x: x / soma)
        x1 = sm.add_constant(x1)

        logit = Logit(endog=y, exog=x1, missing="drop")
        result = logit.fit(method="newton", maxiter=50, full_output=True, disp=True, tol=1e-6)

        ks_df = pd.DataFrame(columns=["target", "pred"])
        ks_df["target"] = y
        ks_df["prob"] = result.predict(x1)

        params = []
        params2 = []
        final = pd.DataFrame()

        odds = np.exp(float(result.params.values[1]) * 100000 / soma)
        sumar = result.summary2()
        ll = float(sumar.tables[0][3][3])
        llnull = float(sumar.tables[0][3][4])
        llr = 2 * (ll - llnull)
        llrp = float(sumar.tables[0][3][5])
        params = pd.DataFrame(
            sumar.tables[1].loc["num_donation_ammount"].values[0:4],
            index=["Coeficiente", "Desvio Padrão", "z", "p-valor"],
        ).T
        params["beta"] = r"$\beta_1$"
        params["Log-Likelihood"] = ll
        params["LL-Null"] = llnull
        params["LLR"] = llr
        params["LLR p"] = llrp
        params["Odds ratio"] = odds
        params["Valor (R\$)"] = soma
        params["N"] = len(x1)
        params["n"] = sum(y)
        params["KS"] = stats.ks_2samp(ks_df.loc[ks_df["target"] == 1, "prob"], ks_df.loc[ks_df["target"] == 0, "prob"])[
            0
        ]
        params["KS p"] = stats.ks_2samp(
            ks_df.loc[ks_df["target"] == 1, "prob"], ks_df.loc[ks_df["target"] == 0, "prob"]
        )[1]
        final = final.append(params)
        params2 = pd.DataFrame(
            sumar.tables[1].loc["const"].values[0:4], index=["Coeficiente", "Desvio Padrão", "z", "p-valor"]
        ).T
        params2["beta"] = r"$\beta_0$"
        params2["Log-Likelihood"] = ll
        params2["LL-Null"] = llnull
        params2["LLR"] = llr
        params2["LLR p"] = llrp
        params2["Odds ratio"] = odds
        params2["Valor (R\$)"] = soma
        params2["N"] = len(x1)
        params2["n"] = sum(y)
        params2["KS"] = stats.ks_2samp(
            ks_df.loc[ks_df["target"] == 1, "prob"], ks_df.loc[ks_df["target"] == 0, "prob"]
        )[0]
        params2["KS p"] = stats.ks_2samp(
            ks_df.loc[ks_df["target"] == 1, "prob"], ks_df.loc[ks_df["target"] == 0, "prob"]
        )[1]
        final = final.append(params2)

        format_mapping = {
            "LL-Null": "{:.2f}".format,
            "Log-Likelihood": "{:.2f}".format,
            "LLR": "{:.2f}".format,
            "LLR p": "{:.2f}".format,
            "Odds ratio": "{:.2f}".format,
            "Valor (R\$)": "{:.0f}".format,
            "N": "{:.0f}".format,
            "n": "{:.0f}".format,
            "KS": "{:.2f}".format,
            "KS p": "{:.2f}".format,
        }
        for key, value in format_mapping.items():
            final[key] = final[key].apply(value)

        final_table = final.set_index(
            ["Log-Likelihood", "LL-Null", "LLR", "LLR p", "KS", "KS p", "Odds ratio", "Valor (R\$)", "N", "n", "beta"]
        )

        final_table.to_csv(
            self.table_path / f"brazil_{self.year}_{self.role}_donations_logistic_regression.csv"
        )

        with open(self.table_path / f"brazil_{self.year}_{self.role}_donations_logistic_regression.tex", "w") as tf:
            tf.write(final_table.to_latex(index=True, escape=False, bold_rows=True, float_format="{:0.2f}".format))

        params = []
        params2 = []
        final = pd.DataFrame()

        estados = g_merged_receitas["cat_federative_unity"].unique()
        for uf in estados:
            print(f"Running LR for {uf}")
            try:
                x1 = raw_data.loc[raw_data["cat_federative_unity"] == uf, "num_donation_ammount"]
                y = raw_data.loc[raw_data["cat_federative_unity"] == uf, "cat_sit_tot_turn_description"].apply(
                    lambda x: 1 if x in ["ELEITO POR QP", "ELEITO POR MÉDIA", "ELEITO"] else 0
                )

                soma = x1.sum()

                x1 = x1.apply(lambda x: x / soma)
                x1 = sm.add_constant(x1)

                logit = Logit(endog=y, exog=x1, missing="drop")
                result = logit.fit(method="newton", maxiter=500, full_output=True, disp=True, tol=1e-4)

                ks_df = pd.DataFrame(columns=["target", "pred"])
                ks_df["target"] = y
                ks_df["prob"] = result.predict(x1)

                odds = np.exp(float(result.params.values[1]) * 100000 / soma)
                sumar = result.summary2()
                ll = float(sumar.tables[0][3][3])
                llnull = float(sumar.tables[0][3][4])
                llr = 2 * (ll - llnull)
                llrp = float(sumar.tables[0][3][5])
                params = pd.DataFrame(
                    sumar.tables[1].loc["num_donation_ammount"].values[0:4],
                    index=["Coeficiente", "Desvio Padrão", "z", "p-valor"],
                ).T
                params["beta"] = r"$\beta_1$"
                params["Log-Likelihood"] = ll
                params["LL-Null"] = llnull
                params["LLR"] = llr
                params["LLR p"] = llrp
                params["Odds ratio"] = odds
                params["Estado"] = uf
                params["Valor (R\$)"] = soma
                params["N"] = len(x1)
                params["n"] = sum(y)
                params["KS"] = stats.ks_2samp(
                    ks_df.loc[ks_df["target"] == 1, "prob"], ks_df.loc[ks_df["target"] == 0, "prob"]
                )[0]
                params["KS p"] = stats.ks_2samp(
                    ks_df.loc[ks_df["target"] == 1, "prob"], ks_df.loc[ks_df["target"] == 0, "prob"]
                )[1]
                final = final.append(params)
                params2 = pd.DataFrame(
                    sumar.tables[1].loc["const"].values[0:4], index=["Coeficiente", "Desvio Padrão", "z", "p-valor"]
                ).T
                params2["beta"] = r"$\beta_0$"
                params2["LLR"] = llr
                params2["LLR p"] = llrp
                params2["Log-Likelihood"] = ll
                params2["LL-Null"] = llnull
                params2["Odds ratio"] = odds
                params2["Estado"] = uf
                params2["Valor (R\$)"] = soma
                params2["N"] = len(x1)
                params2["n"] = sum(y)
                params2["KS"] = stats.ks_2samp(
                    ks_df.loc[ks_df["target"] == 1, "prob"], ks_df.loc[ks_df["target"] == 0, "prob"]
                )[0]
                params2["KS p"] = stats.ks_2samp(
                    ks_df.loc[ks_df["target"] == 1, "prob"], ks_df.loc[ks_df["target"] == 0, "prob"]
                )[1]
                final = final.append(params2)
            except:
                self.logger.error(f"Problem found in {uf}")

        # fitTab = final.sort_values(by="Estado").set_index("Estado").reset_index()
        for key, value in format_mapping.items():
            final[key] = final[key].apply(value)
        self.fitTab = final.set_index(
            [
                "Estado",
                "Log-Likelihood",
                "LL-Null",
                "LLR",
                "LLR p",
                "KS",
                "KS p",
                "Odds ratio",
                "Valor (R\$)",
                "N",
                "n",
                "beta",
            ]
        )

        self.fitTab.to_csv(self.table_path / f"brazil_{self.year}_{self.role}_donations_logistic_regression_by_state.csv")
        
        with open(
            self.table_path / f"brazil_{self.year}_{self.role}_donations_logistic_regression_by_state.tex", "w"
        ) as tf:
            tf.write(self.fitTab.to_latex(index=True, escape=False, bold_rows=True, float_format="{:0.2f}".format))

    def generative_model(self):
        if self.should_fit == True:
            self.fit(self.receitas)

        self.write_latex_tables()
        self.save_figures()
