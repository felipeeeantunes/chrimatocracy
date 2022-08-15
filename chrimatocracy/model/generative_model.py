import csv
import logging
import os
import subprocess
from pathlib import Path
from shutil import ExecError

import jinja2
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

mpl.use("pgf")
pgf_with_pdflatex = {
    "pgf.texsystem": "pdflatex",
    "pgf.preamble": [
        r"\usepackage[utf8x]{inputenc}",
        r"\usepackage[T1]{fontenc}",
        r"\usepackage{cmbright}",
    ],
}
mpl.rcParams.update(pgf_with_pdflatex)

pd.options.display.float_format = "{:.2f}".format
rc = {
    "savefig.dpi": 75,
    "figure.autolayout": False,
    "figure.figsize": [12, 8],
    "axes.labelsize": 18,
    "axes.titlesize": 18,
    "font.size": 14,
    "lines.linewidth": 2.0,
    "lines.markersize": 8,
    "legend.fontsize": 16,
    "xtick.labelsize": 16,
    "ytick.labelsize": 16,
}

sns.set(style="whitegrid", rc=rc)


class GenerativeModel:
    def __init__(
        self,
        year: int,
        role: str,
        data_path: Path,
        table_path: Path,
        figure_path: Path,
        logger=logging.Logger,
    ):

        self.year = year
        self.role = role
        self.data_path = data_path
        self.figure_path = figure_path
        self.table_path = table_path
        self.logger = logger

        self.templates_dir = Path(__file__).parent.parent.resolve() / "templates"

        csv_path = Path(table_path) / "csv"
        if not os.path.exists(csv_path):
            os.makedirs(csv_path)

        tex_path = Path(table_path) / "tex"
        if not os.path.exists(tex_path):
            os.makedirs(tex_path)

        pdf_path = Path(table_path) / "pdf"
        if not os.path.exists(pdf_path):
            os.makedirs(pdf_path)

        figs_tex_path = Path(figure_path) / "tex"
        if not os.path.exists(figs_tex_path):
            os.makedirs(figs_tex_path)

        figs_pdf_path = Path(figure_path) / "pdf"
        if not os.path.exists(figs_pdf_path):
            os.makedirs(figs_pdf_path)

    @staticmethod
    def Fun(X, gamma, eta0):
        delta = np.log(0.01)
        etaMax = np.log(max(X.values))

        A = 1 + (eta0 / (etaMax - delta)) ** gamma
        B = 1 + (eta0 / (np.log(X.values) - delta)) ** gamma

        f = A / B
        return f

    @staticmethod
    def drand_distr(gamma, csi0, csim=100.0, delt=0.0):
        # This generates a random number with distribution according to eq 6 in the manuscript
        F = np.random.uniform(0, 1)
        bla = csi0 / (csim - delt)
        blabla = F / (bla**gamma + 1 - F)

        return np.exp(csi0 * blabla ** (1.0 / gamma) + delt)

    @staticmethod
    def func(x, gamma, csi0, csim=100.0, delt=0.0):
        # this is F(x) in eq. 5
        # cumulative distribution
        num1 = np.log(x) - delt
        num2 = (csi0 / num1) ** gamma
        num3 = (csi0 / (csim - delt)) ** gamma
        term2 = 1.0 + num3
        term3 = 1.0 / (num2 + 1.0)
        return term2 * term3

    @staticmethod
    def func2(x, gamma, csi0, csim=100.0, delt=0.0):
        # this is f(x) in eq. 6
        # distribution
        num1 = np.log(x) - delt
        num2 = (csi0 / num1) ** gamma
        num3 = (csi0 / (csim - delt)) ** gamma
        term1 = gamma / x
        term2 = 1.0 + num3
        term3 = num2 / ((num2 + 1.0) ** 2)
        term3 /= num1
        return term1 * term2 * term3

    @staticmethod
    def lkhd_distr(nums, gamma, csi0, csim, delt):
        # This evaluates lnL and its gradient
        sum1 = 0.0
        sum2 = 0.0
        sum3 = 0.0
        sum4 = 0.0
        sum5 = 0.0
        sum6 = 0.0
        N = len(nums)
        num1 = csi0 / (csim - delt)
        num1g = num1**gamma
        termb = num1g / (1.0 + num1g)
        for ele in nums:
            num2 = np.log(ele)
            num3 = num2 - delt
            num4 = csi0 / (num3)
            num4g = num4**gamma
            term1 = num4g / (num4g + 1.0)
            # for lkhd
            sum1 += num2
            sum2 += np.log(num3)
            sum3 += np.log(1.0 + num4g)
            # for dlnldcsi0
            sum4 += term1
            # for dlnldgamma
            sum5 += np.log(num4) * term1
        lkhd = (
            N * (np.log(gamma) + np.log(1.0 + num1g) + gamma * np.log(csi0)) - sum1 - (gamma + 1.0) * sum2 - 2.0 * sum3
        )
        g1 = N * (gamma * termb / csi0 + gamma / csi0) - 2.0 * gamma * sum4 / csi0  # csi0
        g2 = -N * gamma * termb / (csim - delt)  # csim
        g3 = N * (1.0 / gamma + termb * np.log(num1) + np.log(csi0)) - sum2 - 2.0 * sum5  # gamma
        return lkhd, (g1, g2, g3)

    def maxLKHD_distr(self, nums, gamma=3.0, csi0=3.0, csim=100.0, delt=0.01, lamb=1.0, eps=1.0e-4):
        ### This algorithm may be unstable (never finishes) for some small sets of numbers For the ones in the manuscript it is working fine
        # This evaluates parameters for obtaining maximum value of lnL
        lkhd, grad = self.lkhd_distr(nums, gamma, csi0, csim, delt)
        norm = (grad[0] ** 2 + grad[1] ** 2 + grad[2] ** 2) ** 0.5
        ncsi0 = csi0 + lamb * grad[0] / norm
        ncsim = csim  # +lamb*grad[1]/norm
        ngamma = gamma + lamb * grad[2] / norm
        nlkhd, ngrad = self.lkhd_distr(nums, ngamma, ncsi0, ncsim, delt)
        while lamb * norm > eps:
            # print lamb, norm
            if nlkhd > lkhd:  # accepted
                grad = ngrad
                csi0 = ncsi0
                csim = ncsim
                gamma = ngamma
                lkhd = nlkhd
                lamb *= 1.2
            else:
                lamb *= 0.01
            norm = (grad[0] ** 2 + grad[1] ** 2 + grad[2] ** 2) ** 0.5
            ncsi0 = csi0 + lamb * grad[0] / norm
            ncsim = csim  # +lamb*grad[1]/norm
            ngamma = gamma + lamb * grad[2] / norm
            nlkhd, ngrad = self.lkhd_distr(nums, ngamma, ncsi0, ncsim, delt)
        return gamma, csi0, csim

    def fit(self, df, group_column_name, name):

        df_gen = df[[group_column_name, "id_candidate_cpf", "num_donation_ammount"]].copy()

        df_gen.to_csv(
            self.table_path / "csv" / f"brazil_{self.year}_{self.role}_{name}__fit_input.csv",
            index=False,
        )

        with open(self.table_path / "csv" / f"brazil_{self.year}_{self.role}_{name}__fit_parameters.csv", "w") as f1:
            writer = csv.writer(f1, delimiter=",", lineterminator="\n")
            header = ["group", "xmin", "xmax", "gamma", "eta0"]
            writer.writerow(header)

            for g in df_gen[group_column_name].unique():
                cpfs = list(df_gen.loc[df_gen[group_column_name] == g, "id_candidate_cpf"].unique())
                X = df_gen.loc[df_gen[group_column_name] == g, "num_donation_ammount"]
                xmin = min(X.values)
                xmax = max(X.values)
                result = self.maxLKHD_distr(X.values, gamma=1.0, csi0=1, csim=np.log(xmax), delt=np.log(0.005))
                gamma = result[0]
                eta0 = result[1]
                row = [g, xmin, xmax, gamma, eta0]
                writer.writerow(row)
                self.logger.info(
                    f"""'
                            Grupo: {g}\n
                            ---------------------\n
                            xmin: {xmin}\n
                            xmax: {xmax}\n
                            Gamma: {gamma}\n
                            Eta_0: {eta0}\n
                            ---------------------\n
                            """
                )
                for cpf in cpfs:
                    idx = df_gen.loc[
                        (df_gen[group_column_name] == g) & (df_gen["id_candidate_cpf"] == cpf),
                        "num_donation_ammount",
                    ].index
                    for _idx in idx:
                        df_gen.loc[_idx, "num_donation_ammount"] = self.drand_distr(
                            gamma, eta0, csim=np.log(xmax), delt=np.log(0.005)
                        )

            df_gen.to_csv(
                self.table_path / "csv" / f"brazil_{self.year}_{self.role}_{name}__fit_output.csv",
                index=False,
            )

        return df_gen

    def write_latex_tables(self, name):

        # df         = pd.read_csv(self.data_path / f'brazil_{self.year}_{self.role}_{name}__input.csv')
        # df_gen     = pd.read_csv(self.data_path / f'brazil_{self.year}_{self.role}_{name}__output.csv')
        parameters = pd.read_csv(self.table_path / "csv" / f"brazil_{self.year}_{self.role}_{name}__fit_parameters.csv")

        self.logger.info(f"Model Parameters: {parameters}")
        with open(self.table_path / "tex" / f"brazil_{self.year}_{self.role}_{name}__fit_parameters.tex", "w") as tf:
            tf.write(parameters.reset_index().rename(columns={"index": "Group"}).to_latex(index=False))

    compile

    def compile_latex_tables(self):
        template_dir = (self.templates_dir / "tex").as_posix()
        self.logger.debug(f"Latex Template dir: {template_dir}")
        file_dir = (self.table_path / "tex").as_posix()
        self.logger.debug(f"Latex Files dir: {file_dir}")
        output_path = (self.table_path / "pdf").as_posix()
        self.logger.debug(f"Output to PDF Files dir: {output_path}")

        latex_jinja_env = jinja2.Environment(
            block_start_string="\BLOCK{",
            block_end_string="}",
            variable_start_string="\VAR{",
            variable_end_string="}",
            comment_start_string="\#{",
            comment_end_string="}",
            line_statement_prefix="%-",
            line_comment_prefix="%#",
            trim_blocks=True,
            autoescape=False,
            loader=jinja2.FileSystemLoader(template_dir),
        )

        template = latex_jinja_env.get_template("tables.sty")

        # giving file extension
        ext = ".tex"
        files_list = []
        # iterating over all files
        for files in os.listdir(file_dir):
            if files.endswith(ext):
                files_list.append(files)  # printing file name of desired extension
            else:
                continue

        # template = latex_jinja_env.get_template((file_dir / 'template.jinja').as_posix())
        for file in files_list:
            tex_file_path = (self.table_path / "tex" / f"{file}").as_posix()
            tmp_tex_file_path = (self.table_path / "tex" / f"{file}".replace("tex", "tmp")).as_posix()
            document = template.render(place=tex_file_path)
            with open(tmp_tex_file_path, "w") as output:
                output.write(document)
            x = subprocess.call(f"pdflatex -output-directory={output_path} {tmp_tex_file_path}")
            if x != 0:
                print("Exit-code not 0 for " + file + ", check Code!")

        os.system(f"del {(self.table_path / 'pdf' / '*.log')}")
        os.system(f"del {(self.table_path / 'pdf' / '*.aux')}")
        os.system(f"del {(self.table_path / 'tex' / '*.tmp')}")

    def save_figures(self, group_column_name, name, group_list=None, show=False):

        df = pd.read_csv(self.table_path / "csv" / f"brazil_{self.year}_{self.role}_{name}__fit_input.csv")
        df_gen = pd.read_csv(self.table_path / "csv" / f"brazil_{self.year}_{self.role}_{name}__fit_output.csv")
        group_list = group_list[:28] if group_list is not None else df[group_column_name].unique()[:28]
        _, axes = plt.subplots(
            nrows=7, ncols=4, sharex=True, sharey=True, figsize=(24, 32), gridspec_kw={"hspace": 0.05, "wspace": 0.05}
        )
        axes_list = [item for sublist in axes for item in sublist]

        for g in group_list:
            ax = axes_list.pop(0)
            H2, X2 = np.histogram(np.log(df.loc[df[group_column_name] == g, "num_donation_ammount"]), density=True)
            dx2 = X2[1] - X2[0]
            F2 = np.cumsum(H2) * dx2
            ax.plot(X2[1:], F2, label=f"Data from {str(g)}")
            ax.fill_between(X2[1:], F2, alpha=0.2)

            H2, X2 = np.histogram(
                np.log(df_gen.loc[df_gen[group_column_name] == g, "num_donation_ammount"]), density=True
            )
            dx2 = X2[1] - X2[0]
            F2 = np.cumsum(H2) * dx2
            ax.plot(X2[1:], F2, color="r", label=f"Model for {str(g)}")

            ax.legend()
            # ax.set_title('Group Index: '+ str(g), x=0.1, y=0.1)
            ax.tick_params(which="both", bottom=False, left=False, right=False, top=False)
            # ax.grid(linewidth=0.1)
            # ax.set_xlabel('$\ln(x)$')
            # ax.set_ylabel('$F(x)$')
            ax.set_xlim((-2, 12))
            ax.set_xticks((-2, 0, 2, 4, 6, 8, 10, 12))
            ax.spines["left"].set_visible(False)
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)
            ax.spines["bottom"].set_visible(False)

        for ax in axes.flat:
            ax.set(xlabel="$\ln(x)$", ylabel="$F(x)$")
        for ax in axes.flat:
            ax.label_outer()

        for ax in axes_list:
            ax.remove()

        # plt.tight_layout()
        # plt.subplots_adjust(hspace=0.1, vspace=0.0)

        plt.savefig(self.figure_path / "tex" / f"brazil_{self.year}_{self.role}_{name}__cdf.pgf")
        self.logger.info(f"Figure 'brazil_{self.year}_{self.role}_{name}__cdf.pgf' saved.")

        if show == True:
            plt.show()

    def compile_latex_figures(self):
        template_dir = (self.templates_dir / "tex").as_posix()
        self.logger.debug(f"Latex Template dir: {template_dir}")
        file_dir = (self.figure_path / "tex").as_posix()
        self.logger.debug(f"Latex Files dir: {file_dir}")
        output_path = (self.figure_path / "pdf").as_posix()
        self.logger.debug(f"Output to PDF Files dir: {output_path}")

        latex_jinja_env = jinja2.Environment(
            block_start_string="\BLOCK{",
            block_end_string="}",
            variable_start_string="\VAR{",
            variable_end_string="}",
            comment_start_string="\#{",
            comment_end_string="}",
            line_statement_prefix="%-",
            line_comment_prefix="%#",
            trim_blocks=True,
            autoescape=False,
            loader=jinja2.FileSystemLoader(template_dir),
        )

        template = latex_jinja_env.get_template("figures.sty")

        # giving file extension
        ext = ".pgf"
        files_list = []
        # iterating over all files
        for files in os.listdir(file_dir):
            if files.endswith(ext):
                files_list.append(files)  # printing file name of desired extension
            else:
                continue

        # template = latex_jinja_env.get_template((file_dir / 'template.jinja').as_posix())
        for file in files_list:
            tex_file_path = (self.figure_path / "tex" / f"{file}").as_posix()
            tmp_tex_file_path = (self.figure_path / "tex" / f"{file}".replace("tex", "tmp")).as_posix()
            document = template.render(place=tex_file_path)
            with open(tmp_tex_file_path, "w") as output:
                output.write(document)
            x = subprocess.call(f"pdflatex -output-directory={output_path} {tmp_tex_file_path}")
            if x != 0:
                print("Exit-code not 0 for " + file + ", check Code!")

        os.system(f"del {(self.figure_path / 'pdf' / '*.log')}")
        os.system(f"del {(self.figure_path / 'pdf' / '*.aux')}")
        os.system(f"del {(self.figure_path / 'tex' / '*.tmp')}")
