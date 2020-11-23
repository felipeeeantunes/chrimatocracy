
from pathlib import Path
import sys
import csv
import pandas as pd
import numpy as np
import statsmodels.api as sm
from statsmodels.discrete.discrete_model import Logit
import matplotlib as mpl
import matplotlib.pyplot as plt

from scipy import stats
from colorama import Fore

from matplotlib import ticker
mpl.use("pgf")
pgf_with_pdflatex = {
    "pgf.texsystem": "pdflatex",
    "pgf.preamble":  "\n".join([
        r"\usepackage[utf8x]{inputenc}",
        r"\usepackage[T1]{fontenc}",
        r"\usepackage{cmbright}",
    ])
}
mpl.rcParams.update(pgf_with_pdflatex)

parent_dir = Path().cwd()
print((Path(parent_dir) / 'assets/'))
sys.path.append(str(Path(parent_dir) / 'assets/'))
from generative_model import BenfordGenerativeModel
import benford as bf


class Chrimatocracy(BenfordGenerativeModel):
    def __init__(self,
                 role,
                 data_path,
                 table_path,
                 figure_path,
                 group,
                 should_fit=True):

        super().__init__(
            role,
            data_path,
            table_path,
            figure_path,
            group)

        self.data_types = {
            'id_accountant_cnpj': str,
            'id_candidate_cpf': str,
            'id_donator_cpf_cnpj': str,
            'id_original_donator_cpf_cnpj': str,
            'id_donator_effective_cpf_cnpj': str
        }

        self.should_fit = should_fit

    def load_donations(self):
        self.receitas = pd.read_csv(
            self.data_path / f'brazil_2014_{self.role}_candidates_donations.csv', dtype=self.data_types, low_memory=False)
        self.merged_receitas = pd.read_csv(
            self.data_path / f"brazil_2014_{self.role}_accepted_candidates_votes_with_candidate_info_and_donations.csv", dtype=self.data_types, low_memory=False)

    def benford_plot(self):
        self.all_digits = bf.read_numbers(
            self.receitas['num_donation_ammount'])
        self.all_probs = bf.find_probabilities(self.all_digits)

        x2, _ = bf.find_x2(pd.Series(self.all_digits))

        width = 0.2

        indx = np.arange(1, len(self.all_probs) + 1)
        benford = [np.log10(1 + (1.0 / d)) for d in indx]

        plt.bar(indx, benford, width, color='r', label="Lei de Benford",)
        plt.bar(indx+width, self.all_probs, width, color='#1DACD6',
                label=r'Doações($\chi^2$''='+str(round(x2, 2))+')')
        plt.yscale('log')
        plt.xscale('log')
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
        plt.savefig(
            figure_path / f"brazil_2014_{self.role}_benford_distribution.pgf")
        #plt.show()

    def lr_table(self):

        g_key = ['id_candidate_cpf', 'cat_federative_unity',
                 'int_election_turn', 'cat_sit_tot_turn_description']
        g_merged_receitas = self.merged_receitas.groupby(g_key).agg(
            {"int_number_of_votes": lambda x: sum(x)/len(x), 'num_donation_ammount': 'sum'}).reset_index()

        r_key = ['id_candidate_cpf',
                 'cat_federative_unity', 'int_election_turn']
        rg_key = ['id_candidate_cpf', 'cat_federative_unity']
        
        raw_data = g_merged_receitas.sort_values(by=r_key).groupby(rg_key).agg(
            {'cat_sit_tot_turn_description': lambda x: np.array(x)[-1], 'num_donation_ammount': 'sum'}).reset_index()

        x1 =  raw_data['num_donation_ammount']
        y  =  raw_data['cat_sit_tot_turn_description'].apply(lambda x: 1 if x in ['ELEITO POR QP', 'ELEITO POR MÉDIA', 'ELEITO'] else 0)
    
        
        soma = x1.sum()

        x1 = x1.apply(lambda x: x/soma)
        x1 = sm.add_constant(x1)

        logit = Logit(endog=y, exog=x1, missing='drop')
        result = logit.fit(method='newton', maxiter=50,
                        full_output=True, disp=True, tol=1e-6)

        

        ks_df = pd.DataFrame(columns=['target', 'pred'])
        ks_df['target'] = y
        ks_df['prob'] = result.predict(x1)

        params = []
        params2 = []
        final = pd.DataFrame()

        odds = np.exp(float(result.params.values[1]) * 100000/soma)
        sumar = result.summary2()
        llrp = float(sumar.tables[0][3][5]) 
        params = pd.DataFrame(sumar.tables[1].loc["num_donation_ammount"].values[0:4], index=[
                                "Coeficiente", 'Desvio Padrão', 'z', "p-valor"]).T
        params["beta"] = r'$\beta_1$'
        params["LLR p"] = llrp
        params["Odds ratio"] = odds
        params["Valor"] = soma
        params["N"] = len(x1)
        params["n"] = sum(y)
        params["KS"] = stats.ks_2samp(ks_df.loc[ks_df['target'] == 1, 'prob'], ks_df.loc[ks_df['target'] == 0, 'prob'])[0]
        params["KS p"] = stats.ks_2samp(ks_df.loc[ks_df['target'] == 1, 'prob'], ks_df.loc[ks_df['target'] == 0, 'prob'])[1]
        final = final.append(params)
        params2 = pd.DataFrame(sumar.tables[1].loc["const"].values[0:4], index=[
                            "Coeficiente", 'Desvio Padrão', 'z', "p-valor"]).T
        params2["beta"] = r'$\beta_0$'
        params2["LLR p"] = llrp
        params2["Odds ratio"] = odds
        params2["Valor"] = soma
        params2["N"] = len(x1)
        params2["n"] = sum(y)
        params2["KS"] = stats.ks_2samp(ks_df.loc[ks_df['target'] == 1, 'prob'], ks_df.loc[ks_df['target'] == 0, 'prob'])[0]
        params2["KS p"] = stats.ks_2samp(ks_df.loc[ks_df['target'] == 1, 'prob'], ks_df.loc[ks_df['target'] == 0, 'prob'])[1]
        final = final.append(params2)

        format_mapping={
                "LLR p": '{:0.2g}'.format,
                "Odds ratio": '{:.2f}'.format,
                "Valor": '{:0.2g}'.format,
                "N":'{:.0f}'.format,
                "n": '{:.0f}'.format,
                "KS": '{:.2f}'.format,
                "KS p":'{:0.2g}'.format
            }
        for key, value in format_mapping.items():
             final[key] = final[key].apply(value)


        final_table = final.set_index(
            ["LLR p", "KS", "KS p", "Odds ratio", "Valor", "N", "n", "beta"])
        


        with open(table_path / f'brazil_2014_{self.role}_donations_logistic_regression.tex', 'w') as tf:
            tf.write(final_table.to_latex(index=True, escape=False, bold_rows=True, float_format="{:0.2f}".format))

        params = []
        params2 = []
        final = pd.DataFrame()
        

        estados = g_merged_receitas['cat_federative_unity'].unique()
        for uf in estados:
            print(f"Running LR for {uf}")
            try:
                x1 = raw_data.loc[raw_data['cat_federative_unity'] == uf,'num_donation_ammount']
                y  =  raw_data.loc[raw_data['cat_federative_unity'] == uf, 'cat_sit_tot_turn_description'].apply(lambda x: 1 if x in ['ELEITO POR QP', 'ELEITO POR MÉDIA', 'ELEITO'] else 0)
                
                soma = x1.sum()

                x1 = x1.apply(lambda x: x/soma)
                x1 = sm.add_constant(x1)

                logit = Logit(endog=y, exog=x1, missing='drop')
                result = logit.fit(method='newton', maxiter=500,
                                full_output=True, disp=True, tol=1e-4)
                
                ks_df = pd.DataFrame(columns=['target', 'pred'])
                ks_df['target'] = y
                ks_df['prob'] = result.predict(x1)

                odds = np.exp(float(result.params.values[1]) * 100000/soma)
                sumar = result.summary2()
                llrp = float(sumar.tables[0][3][5])
                params = pd.DataFrame(sumar.tables[1].loc["num_donation_ammount"].values[0:4], index=[
                                     "Coeficiente", 'Desvio Padrão', 'z', "p-valor"]).T
                params["beta"] = r'$\beta_1$'
                params["LLR p"] = llrp
                params["Odds ratio"] = odds
                params["Estado"] = uf
                params["Valor"] = soma
                params["N"] = len(x1)
                params["n"] = sum(y)
                params["KS"] = stats.ks_2samp(ks_df.loc[ks_df['target'] == 1, 'prob'], ks_df.loc[ks_df['target'] == 0, 'prob'])[0]
                params["KS p"] = stats.ks_2samp(ks_df.loc[ks_df['target'] == 1, 'prob'], ks_df.loc[ks_df['target'] == 0, 'prob'])[1]
                final = final.append(params)
                params2 = pd.DataFrame(sumar.tables[1].loc["const"].values[0:4], index=[
                                    "Coeficiente", 'Desvio Padrão', 'z', "p-valor"]).T
                params2["beta"] = r'$\beta_0$'
                params2["LLR p"] = llrp
                params2["Odds ratio"] = odds
                params2["Estado"] = uf
                params2["Valor"] = soma
                params2["N"] = len(x1)
                params2["n"] = sum(y)
                params2["KS"] = stats.ks_2samp(ks_df.loc[ks_df['target'] == 1, 'prob'], ks_df.loc[ks_df['target'] == 0, 'prob'])[0]
                params2["KS p"] = stats.ks_2samp(ks_df.loc[ks_df['target'] == 1, 'prob'], ks_df.loc[ks_df['target'] == 0, 'prob'])[1]
                final = final.append(params2)
            except:
                print(f"Problem found in {uf}")
        # %%
        #fitTab = final.sort_values(by="Estado").set_index("Estado").reset_index()
        for key, value in format_mapping.items():
             final[key] = final[key].apply(value)
        self.fitTab = final.set_index(
            ["Estado", "LLR p", "KS", "KS p", "Odds ratio", "Valor", "N", "n", "beta"])

        with open(table_path / f'brazil_2014_{self.role}_donations_logistic_regression_by_state.tex', 'w') as tf:
            tf.write(self.fitTab.to_latex(index=True, escape=False, bold_rows=True, float_format="{:0.2f}".format))

    # %% [markdown]
    # Vamos obter a odds ratio exponenciando o coeficiente obtido. Esse parâmetro nos diz como o incremento ou decremento
    # de uma unidade na variável explicativa afeta as chances do candadito ser eleito. Podemos desfazer a normalização adotada para entender a importância do dinheiro para a variável resposta.
    # %% [markdown]
    # ---

    # %%
    # for j,i in zip(final['Estado'].unique(), final['Odds ratio'].unique()):
    #     print("Cada R$ 100.000 aumenta as chances de um deputado do ",j," ser eleito em", '%.2f'% float((i-1)*100), "%")

    # def ks(self):
    #     data = self.ks_df
    #     target = 'target'
    #     prob = 'prob'
    #     data['target0'] = 1 - data[target]
    #     data['bucket'] = pd.qcut(data[prob], 10, duplicates='drop')
    #     grouped = data.groupby('bucket', as_index=False)
    #     kstable = pd.DataFrame()
    #     kstable['min_prob'] = grouped.min()[prob]
    #     kstable['max_prob'] = grouped.max()[prob]
    #     kstable['events'] = grouped.sum()[target]
    #     kstable['nonevents'] = grouped.sum()['target0']
    #     kstable = kstable.sort_values(
    #         by="min_prob", ascending=False).reset_index(drop=True)
    #     kstable['event_rate'] = (
    #         kstable.events / data[target].sum()).apply('{0:.2%}'.format)
    #     kstable['nonevent_rate'] = (
    #         kstable.nonevents / data['target0'].sum()).apply('{0:.2%}'.format)
    #     kstable['cum_eventrate'] = (
    #         kstable.events / data[target].sum()).cumsum()
    #     kstable['cum_noneventrate'] = (
    #         kstable.nonevents / data['target0'].sum()).cumsum()
    #     kstable['KS'] = np.round(
    #         kstable['cum_eventrate']-kstable['cum_noneventrate'], 3) * 100

    #     # Formating
    #     kstable['cum_eventrate'] = kstable['cum_eventrate'].apply(
    #         '{0:.2%}'.format)
    #     kstable['cum_noneventrate'] = kstable['cum_noneventrate'].apply(
    #         '{0:.2%}'.format)
        
    #     kstable.index = range(1, kstable.shape[0]+1)
    #     kstable.index.rename('Decile', inplace=True)
    #     pd.set_option('display.max_columns', 9)
    #     print(kstable)

    #     # Display KS
    #     print(Fore.RED + "KS is " + str(max(kstable['KS']))+"%" + " at decile " + str(
    #         (kstable.index[kstable['KS'] == max(kstable['KS'])][0])))
        
        
        
    #     return(kstable)


    def generative_model(self):
        if self.should_fit == True:
            self.fit(self.receitas, self.role)

        self.write_latex_tables()
        self.save_figures()


if __name__ == "__main__":
    data_path = Path(parent_dir) / 'data/'
    table_path = Path(parent_dir) / 'tables/'
    figure_path = Path(parent_dir) / 'figures/'

    # [], "senator", "federal_deputy", "state_deputy", "district_deputy", "president"]
    roles = ["state_deputy", "federal_deputy"]

    for role in roles:
        print(f"Generating Chrimatocracy Analysis table for {role}.\n")
        chrimatocracy = Chrimatocracy(role=role,
                                      data_path=data_path,
                                      table_path=table_path,
                                      figure_path=figure_path,
                                      group='cat_federative_unity',
                                      should_fit=False)

        chrimatocracy.load_donations()
        chrimatocracy.benford_plot()
        chrimatocracy.lr_table()
        #chrimatocracy.ks()
        chrimatocracy.generative_model()
