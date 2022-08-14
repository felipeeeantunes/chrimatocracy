import itertools
import logging
from collections import defaultdict
from pathlib import Path
from shutil import ExecError
import igraph
import leidenalg as louvain
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import ticker
from operator import itemgetter
from random import randint
from chrimatocracy.model import GenerativeModel
from chrimatocracy.utils import assignmentArray_to_lists
from chrimatocracy.utils import benford as bf
from chrimatocracy.utils import draw_adjacency_matrix

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


class Network(GenerativeModel):
    def __init__(
        self,
        year: int,
        role: str,
        state: str,
        data_path: Path,
        table_path: Path,
        figure_path: Path,
        community_column_name: str,
        min_obs: int = 100,
        should_fit: bool = True,
        benford_log_scale: bool = True,
        logger: logging.Logger = None,
    ):

        super().__init__(year, role, data_path, table_path, figure_path, logger)

        self.should_fit = should_fit
        self.benford_log_scale = benford_log_scale
        self.state = state
        self.min_obs = min_obs
        self.logger = logger
        self.path_prefix = f"brazil_{self.year}_{self.role}_{self.state}"
        self.community_column_name = community_column_name
      

    def load_donations_to_candidates(self) -> pd.DataFrame:
        self.logger.info("Loading donations to candidates DataFrame from prepared data...")
        data_types = {
            "id_accountant_cnpj": str,
            "id_candidate_cpf": str,
            "id_donator_cpf_cnpj": str,
            "id_original_donator_cpf_cnpj": str,
            "id_donator_effective_cpf_cnpj": str,
        }

        candidatos = pd.read_csv(
            self.data_path / f"brazil_{self.year}_{self.role}_candidates_donations.csv",
            low_memory=False,
            dtype=data_types,
        )

        df = candidatos[
            (candidatos["id_donator_effective_cpf_cnpj"].isnull() == False)
            & (candidatos["cat_federative_unity"] == self.state)
        ]

        self.logger.info("Donations to candidates DataFrame loaded.")
        return df

    def create_adj_matrix(self, df: pd.DataFrame=None, write: bool=True) -> pd.DataFrame:

        idx = df.id_candidate_cpf.unique()
        adj = pd.DataFrame(0, index=idx, columns=idx)

        for donator in idx:
            rec = df.loc[df.id_donator_effective_cpf_cnpj == donator, "id_candidate_cpf"].unique()
            cpf_pairs = list(itertools.combinations(rec, 2))
            for cpf1, cpf2 in cpf_pairs:
                adj.loc[cpf1, cpf2] += 1.0

        adj.reset_index(inplace=True)
        if write:
            adj.to_csv(self.table_path / "csv" / f"{self.path_prefix}__adj_matrix.csv", index=False)
            self.logger.info(f"Adjacency matrix saved: {self.table_path}csv/{self.path_prefix}__adj_matrix.csv")

        return adj.set_index("index")

    def detect_communities(self, adj: pd.DataFrame=None,  write=True, read: bool=False):
        
        if read:
            self.logger.info(f"Loading previous created adjacency matrix from {self.table_path}csv/{self.path_prefix}__adj_matrix.csv")
            adj = pd.read_csv(
                    self.table_path / "csv" / f"{self.path_prefix}__adj_matrix.csv",
                    converters={"index": lambda x: str(x).zfill(11)},
                ).set_index("index")
            self.logger.info(f"Adjacency matrix loaded.")
        elif adj is None:
            raise ValueError("You should read a previous created adjacency matrix or provide a DataFrame.")

        self.logger.info("Creating Graph from Adjacency Matrix...")
        # Get the values as np.array, it's more convenenient.
        A = adj.values
        # Create graph, A.astype(bool).tolist() or (A / A).tolist() can also be used.
        G = igraph.Graph.Adjacency((A > 0).tolist())
        # Add edge weights and node labels.
        G.es["weight"] = A[A.nonzero()]
        G.vs["label"] = adj.index  # or a.index/a.columns
        
        self.logger.info("Graph created. Network summary:")
        g_n_vertices = G.vcount()
        self.logger.info(f"Number of vertices (nodes): {g_n_vertices}")
        g_n_edges = G.ecount()
        self.logger.info(f"Number of edges: {g_n_edges}")
        g_avg_degree = np.mean(G.degree())
        self.logger.info(f"Average Degree: {g_avg_degree}")
        network_summary = pd.DataFrame.from_dict(
            {
                "Number of vertices (nodes)": [g_n_vertices],
                "Number of edges": [g_n_edges],
                "Average Degree": [g_avg_degree],
            }
        )
        network_summary.to_csv(self.table_path / "csv" / f"{self.path_prefix}__network_summary.csv")
        self.logger.info(f"Saved in CSV format at: {self.table_path}csv/{self.path_prefix}__nextwork_summary.csv")
        with open(self.table_path / "tex" / f"{self.path_prefix}_network_summary.tex", "w") as tf:
            tf.write(network_summary.to_latex(index=False))
        self.logger.info(f"Summary in TeX format at: {self.table_path}tex/{self.path_prefix}__nextwork_summary.csv")


        self.logger.info(f"Running community detection algorithm...")
        part = louvain.find_partition(G, louvain.ModularityVertexPartition, weights="weight")
        G.vs["community"] = part.membership
        self.logger.info(f"Communities identified.")
        p_quality = part.quality()
        self.logger.info(f"Quality:  {p_quality}")
        p_modularity = part.modularity
        self.logger.info(f"Modularity:  {p_modularity}")
        p_summary = part.summary()
        self.logger.info(f"Partitions Summary: {p_summary}")
        partitions_summary = pd.DataFrame.from_dict(
            {"Partitions Summary": [p_summary], "Quality": [p_quality], "Modularity": [p_modularity]}
        )
        partitions_summary.to_csv(
            self.table_path / "csv" / f"{self.path_prefix}__partition_summary.csv"
        )
        self.logger.info(f"Saved in CSV format at: {self.table_path}csv/{self.path_prefix}__partition_summary.csv")
        with open(self.table_path / "tex" / f"{self.path_prefix}_partitions_summary.tex", "w") as tf:
            tf.write(partitions_summary.to_latex(index=False))
        self.logger.info(f"Saved in TeX format at: {self.table_path}tex/{self.path_prefix}__partition_summary.tex")
        
    
        return G
        
    def append_communities(self, G, df: pd.DataFrame=None, write=True, read=False): 
        if read:
            self.logger.info(f"Loading 'donations to candidates' DataFrame from {self.table_path}csv/{self.path_prefix}__communities.csv")
            df = pd.read_csv(
                    self.table_path / "csv" / f"{self.path_prefix}__communities.csv",
                    converters={"index": lambda x: str(x).zfill(11)},
                ).set_index("index")
            self.logger.info(f"Adjacency matrix loaded.")
        elif df is None:
            raise ValueError("You should read a previous created 'donations to candidates' or provide a DataFrame.")

        if G is None:
            raise ValueError("A Graph should be provided.")

        self.logger.info(f"Adding communities to 'donations to candidates' DataFrame...")
        community_dict = dict(zip(G.vs["label"], G.vs["community"]))
        df.loc[:, self.community_column_name] = df.loc[:, "id_candidate_cpf"].map(
            community_dict
        )
        if write:
            self.logger.info(f"Candidates with communities identified saved at: {self.table_path}csv/{self.path_prefix}__communities.csv")
            df.to_csv(
            self.table_path / "csv" / f"{self.path_prefix}__communities.csv", index=False
            )

        return df
        

    def graph_to_matrix(self, G, adj: pd.DataFrame=None, read=True, draw=True):

        if read:
            self.logger.info(f"Loading previous created adjacency matrix from {self.table_path}csv/{self.path_prefix}__adj_matrix.csv")
            adj = pd.read_csv(
                    self.table_path / "csv" / f"{self.path_prefix}__adj_matrix.csv",
                    converters={"index": lambda x: str(x).zfill(11)},
                ).set_index("index")
            self.logger.info(f"Adjacency matrix loaded.")
        elif adj is None:
            raise ValueError("You should read a previous created adjacency matrix or provide a DataFrame.")

        if G is None:
            raise ValueError("A Graph should be provided.")
            #check the possibility to save and load G igraph.Graph.load(self.data_path / f'{self.path_prefix}__network.net')
        
        self.logger.info(f"Adding communities information to the Adjacency Matrix...")
        pr = dict(zip(G.vs["label"], G.pagerank(weights="weight")))
        maxPR = max(pr.values())

        flip_PR = dict(
            zip(
                sorted(dict(zip(G.vs["label"], G.pagerank(weights="weight"))), key=lambda node: pr[node], reverse=True),
                range(0, G.vcount()),
            )
        )

        pr_seq = sorted(pr, reverse=True)  # degree sequence

        
        ## VERIFICAR A LEI DE BENFORD PARA CANDIDATOS A ESQUERDA DO A DIREITA


        community_dict = dict(zip(G.vs["label"], G.vs["community"]))

        # Convert community assignmet dict into list of communities
        comms = defaultdict(list)
        for node_index, comm_id in community_dict.items():
            comms[comm_id].append(node_index)
        comms = comms.values()

        nodes_ordered = [node for comm in comms for node in comm]

        nodes_ordered = []
        for comm in comms:
            nodePR = []
            for node in comm:
                nodePR.append((node, pr[node]))

            nodePR = sorted(nodePR, key=itemgetter(1))
            nodes_ordered.append(nodePR)


        nodes_list = []
        for sublist in nodes_ordered:
            for item in sublist:
                nodes_list.append(item[0])

        
        colors = []

        for i in range(len(comms)):
            colors.append("%06X" % randint(0, 0xFFFFFF))
        
        if draw:
            self.logger.info(f"Drawing Adjacency Matrix figure...")
            draw_adjacency_matrix(
                adj,
                nodes_list,
                comms,
                colors=colors,
                output_file=self.figure_path / "tex" / f"{self.path_prefix}__adj_matrix.pgf",
            )
            self.logger.info(f"Figure saved at: {self.table_path}tex/{self.path_prefix}__adj_matrix.pgf")

        return adj, nodes_list, comms, colors

    def benford_plot(self, df: pd.DataFrame=None, read=False, log_scale=False):
        
        if read:
            self.logger.info(f"Loading 'donations to candidates' DataFrame from {self.table_path}csv/{self.path_prefix}__communities.csv")
            df = pd.read_csv(
                    self.table_path / "csv" / f"{self.path_prefix}__communities.csv",
                    converters={"index": lambda x: str(x).zfill(11)},
                ).set_index("index")
            self.logger.info(f"'donations to candidates' loaded.")
        elif df is None:
            raise ValueError("You should read a previous created 'donations to candidates' or provide a DataFrame.")

        self.logger.info(f"Drawing Benford Law figure...")
        donations = df["num_donation_ammount"]

        self.all_digits = bf.read_numbers(donations)
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

        if log_scale:
            self.logger.info(f"Applying Log Scale for x and y axis...")
            plt.yscale("log")
            plt.xscale("log")
        ax = plt.gca()
        ax.set_xticks(indx)
        ax.set_yticks([0.05, 0.1, 0.2, 0.3])
        ax.get_xaxis().set_major_formatter(ticker.ScalarFormatter())
        ax.get_yaxis().set_major_formatter(ticker.ScalarFormatter())
        plt.title(f"Benford Law for Brazil {self.year} electoral campaing for {self.role} at{self.state}")
        plt.ylabel("Probability")
        plt.grid(False)
        plt.legend()

        plt.savefig(self.figure_path / "tex" / f"{self.path_prefix}_benford_distribution.pgf", bbox_inches='tight')
        self.logger.info(f"Benford Law figure saved: {self.figure_path}tex/{self.path_prefix}__benford_distribution.pgf")

    def benford_table(self, df=None, write=True, read=False):

        if read:
            self.logger.info(f"Loading 'donations to candidates' DataFrame from {self.table_path}csv/{self.path_prefix}__communities.csv")
            df = pd.read_csv(
                    self.table_path / "csv" / f"{self.path_prefix}__communities.csv",
                    converters={"index": lambda x: str(x).zfill(11)},
                ).set_index("index")
            self.logger.info(f"'donations to candidates' loaded.")
        elif df is None:
            raise ValueError("You should read a previous created 'donations to candidates' or provide a DataFrame.")
        
        self.logger.info(f"Generating Benford Law table...")

        ## Benford Law Table
        benford_table = bf.benford_digits_table(df, self.community_column_name)

        benford_table = benford_table.loc[benford_table["N"] >= self.min_obs]

        if write:
            benford_table.to_csv(
            self.table_path / "csv" / f"{self.path_prefix}__benford_law_{self.community_column_name}_gt_{self.min_obs}.csv", index=False
            )
            self.logger.info(f"Saved in CSV format at: {self.table_path}csv/{self.path_prefix}__benford_law_{self.community_column_name}_gt_{self.min_obs}.csv")

            with open(
                self.table_path / "tex"/ f"{self.path_prefix}__benford_law_{self.community_column_name}_gt_{self.min_obs}.tex", "w"
            ) as tf:
                tf.write(benford_table.to_latex(index=False))
            self.logger.info(f"Saved in TeX format at: {self.table_path}tex/{self.path_prefix}__benford_law_{self.community_column_name}_gt_{self.min_obs}.tex")

        return benford_table

    def benford_dirty_stats_table(self, df=None, benford_table=None, write=True, read_df=False, read_benford_table=False, name="20190301"):
        CNEP = pd.read_csv(self.data_path / f"{name}_CNEP.csv")
        CEIS = pd.read_csv(self.data_path / f"{name}_CEIS.csv")

        if read_df:
            self.logger.info(f"Loading 'donations to candidates' DataFrame from {self.table_path}csv/{self.path_prefix}__communities.csv")
            df = pd.read_csv(
                    self.table_path / "csv" / f"{self.path_prefix}__communities.csv",
                    converters={"index": lambda x: str(x).zfill(11)},
                ).set_index("index")
            self.logger.info(f"'donations to candidates' loaded.")
        elif df is None:
            raise ValueError("You should read a previous created 'donations to candidates' or provide a DataFrame.")

        if read_benford_table:
            self.logger.info(f"Loading 'benford table' DataFrame from {self.table_path}csv/{self.path_prefix}__{name}_{self.community_column_name}_gt_{self.min_obs}.csv")
            benford_table = pd.read_csv(
                    self.table_path / "csv" / f"{self.path_prefix}__benford_law_{self.community_column_name}_gt_{self.min_obs}.csv",
                    converters={"index": lambda x: str(x).zfill(11)},
                ).set_index("index")
            self.logger.info(f"'benford table' loaded.")
        elif read_benford_table is None:
            raise ValueError("You should read a previous created 'benford table' or provide a DataFrame.")

        
        cnpjs_dirty = (
            pd.concat([CNEP["CPF OU CNPJ DO SANCIONADO"], CEIS["CPF OU CNPJ DO SANCIONADO"]])
            .apply(lambda x: str(x).zfill(14))
        )  # < contem cpfs tambem
        self.logger.info(f"Number of dirty companies: {cnpjs_dirty.nunique()}")

        
        df.loc[
            :, "id_donator_effective_cpf_cnpj_str"
        ] = df.id_donator_effective_cpf_cnpj.apply(lambda x: str(int(x)).zfill(14))

        
        df.loc[:, "dirty"] = df["id_donator_effective_cpf_cnpj_str"].isin(cnpjs_dirty)

        
        donnors_dirty_list = df.loc[
            df["dirty"], "id_donator_effective_cpf_cnpj_str"
        ].unique()
        self.logger.info(f'Number cpfs/cnpjs dirty donations: {df["dirty"].sum()}')
        
        n_donnors_dirty = df.loc[
            df["dirty"], "id_donator_effective_cpf_cnpj_str"
        ].nunique()
        self.logger.info(f"Number of cpfs/cnpjs dirty donnors: {n_donnors_dirty}")

        
        # contagem de doacoes sujas nas community
        benford_table_dirty_donations = (
            df.groupby(self.community_column_name).agg({"dirty": lambda x: sum(x)}).sort_values(by="dirty", ascending=False)
        )
        benford_table_dirty_donations.columns = ["# Dirty Donations"]
        benford_table_dirty_donations = benford_table.merge(benford_table_dirty_donations, how="left", on=self.community_column_name)

        
        # numero de donnors na comunidade
        donnors = df.groupby(self.community_column_name).agg(
            {"id_donator_effective_cpf_cnpj": lambda x: x.nunique()}
        )
        donnors.columns = ["# Donors"]
        benford_table_donnors = benford_table_dirty_donations.merge(donnors, how="left", on=self.community_column_name)

        
        # numero de doares dirty na comunidade
        donnors_dirty = (
            df.loc[df.id_donator_effective_cpf_cnpj.isin(donnors_dirty_list)]
            .groupby(self.community_column_name)
            .agg({"id_donator_effective_cpf_cnpj": lambda x: int(x.nunique())})
        )
        donnors_dirty.columns = ["# Dirty Donors"]
        benford_table_donnors_dirty = benford_table_donnors.merge(donnors_dirty, how="left", on=self.community_column_name)

        
        # total dirty doado na comunidade
        total_dirty_amount = (
            df.loc[df.id_donator_effective_cpf_cnpj.isin(donnors_dirty_list)]
            .groupby(self.community_column_name)
            .agg({"num_donation_ammount": lambda x: sum(x)})
        )
        total_dirty_amount.columns = ["Total Dirty Amount"]
        benford_table_total_dirty_amount = benford_table_donnors_dirty.merge(total_dirty_amount, how="left", on=self.community_column_name)

        
        # Soma das doacoes na comunidade:
        total_amount = df.groupby(self.community_column_name).agg({"num_donation_ammount": sum})
        total_amount.columns = ["Total Amount"]
        benford_table_total_amount = benford_table_total_dirty_amount.merge(total_amount, how="left", on=self.community_column_name)
        
        # Soma das doacoes na comunidade:
        donnors = df.groupby(self.community_column_name).agg({"id_donator_effective_cpf_cnpj": lambda x: x.nunique()})
        donnors.columns = ["Number of Candidates"]
        benford_table_donnors = benford_table_total_amount.merge(donnors, how="left", on=self.community_column_name)

        if write:
           
            benford_table_donnors.to_csv(
            self.table_path / "csv" / f"{self.path_prefix}__benford_law_{self.community_column_name}_gt_{self.min_obs}.csv", index=False
            )
            self.logger.info(f"Saved in CSV format at: {self.table_path}csv/{self.path_prefix}__benford_law_{self.community_column_name}_gt_{self.min_obs}_dirty_stats.csv")

            with open(
                self.table_path / "tex" / f"{self.path_prefix}__benford_law_{self.community_column_name}_gt_{self.min_obs}_dirty_stats.tex", "w"
            ) as tf:
                tf.write(benford_table_donnors.to_latex(index=False))
            self.logger.info(f"Saved in TeX format at: {self.table_path}tex/{self.path_prefix}__benford_law_{self.community_column_name}_gt_{self.min_obs}_dirty_stats.tex")
        
        return benford_table_donnors



    def generative_model(self, df: pd.DataFrame, should_fit=None, benford_table=None):
        should_fit = should_fit if should_fit is not None else self.should_fit
        self.selected_communities = benford_table[self.community_column_name].unique()
        
        selected_idx = df[self.community_column_name].isin(self.selected_communities)
        df = df.loc[selected_idx]
        
        
        name = f"{self.state}__generative_model_{self.community_column_name}_gt_{self.min_obs}"
        if should_fit == True:
            self.fit(df, group_column_name=self.community_column_name, name = name)
        self.write_latex_tables(name=name)
        self.compile_latex_tables()
        self.save_figures(group_list=self.selected_communities, group_column_name=self.community_column_name, name=name)
        self.compile_latex_figures()
        parameters = pd.read_csv(
            self.table_path / "csv" / f"brazil_{self.year}_{self.role}_{name}__fit_parameters.csv"
        )
        
        benford_table_fit_parameters = benford_table.join(parameters)

        benford_table_fit_parameters.to_csv(
            self.table_path / "csv" / f"{self.path_prefix}__benford_law_{self.community_column_name}_gt_{self.min_obs}_dirty_fit_params.csv", index=False
        )
        self.logger.info(f"Saved in CSV format at: {self.table_path}csv/{self.path_prefix}__benford_law_{self.community_column_name}_gt_{self.min_obs}_dirty_fit_params.csv")

        with open(
            self.table_path / "tex" / f"{self.path_prefix}__benford_law_{self.community_column_name}_gt_{self.min_obs}_dirty_fit_params.tex", "w"
        ) as tf:
            tf.write(benford_table_fit_parameters.to_latex(index=False))
        self.logger.info(f"Saved in TeX format at: {self.table_path}tex/{self.path_prefix}__benford_law_{self.community_column_name}_gt_{self.min_obs}_dirty_fit_params.tex")


        return benford_table_fit_parameters