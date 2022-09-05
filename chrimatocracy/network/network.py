import itertools
import json
import logging
from collections import defaultdict
from operator import itemgetter
from pathlib import Path
from random import randint
from shutil import ExecError

import igraph
import leidenalg as leiden
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import ticker

from chrimatocracy.model import GenerativeModel
from chrimatocracy.utils import assignmentArray_to_lists
from chrimatocracy.utils import benford as bf
from chrimatocracy.utils import (draw_adjacency_matrix, entity_subset,
                                 load_donations_to_candidates,
                                 save_and_log_table, sort_pagerank)

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

ENTITY_DICT = {"donnors": "id_donator_effective_cpf_cnpj", "candidates": "id_candidate_cpf"}


class Network(GenerativeModel):
    def __init__(
        self,
        year: int,
        role: str,
        state: str,
        data_path: Path,
        table_path: Path,
        figure_path: Path,
        json_path: Path,
        main_entity="donnors",
        relationship_entity="candidates",
        main_entity_subset="companies",
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
        self.main_entity = main_entity
        self.relationship_entity = relationship_entity
        self.main_entity_subset = main_entity_subset
        self.community_column_name = 'comm'

        if main_entity not in ENTITY_DICT.keys():
            raise ValueError(f"Main entity should be one of {ENTITY_DICT.keys()}")

    def projected_matrix(
        self,
        df: pd.DataFrame = None,
        write: bool = True,
    ) -> pd.DataFrame:
        self.logger.info(f"Creating Adjacency matrix...")

        main_entity_col = ENTITY_DICT.get(self.main_entity)
        relationship_entity_col = ENTITY_DICT.get(self.relationship_entity)

        df_subset = entity_subset(df, column=main_entity_col, entity=self.main_entity_subset)[[
            main_entity_col, relationship_entity_col
        ]]

        df_merge = df_subset.merge(df_subset, on=relationship_entity_col)

        adj = pd.crosstab(df_merge[f"{main_entity_col}_x"], df_merge[f"{main_entity_col}_y"], dropna=False)

        idx = adj.columns.union(adj.index)
        adj = adj.reindex(index=idx, columns=idx, fill_value=0)
        adj.reset_index(inplace=True)

        if write:
            file_name = f"{self.path_prefix}__adj_matrix.csv"
            save_and_log_table(
                df=adj, file_name=file_name, file_path=self.table_path, formats=["csv"], logger=self.logger
            )

        return adj.set_index("index")

    def _graph_from_matrix(self, adj: pd.DataFrame = None, read: bool = False):
        if read:
            raise NotImplementedError
        elif adj is None:
            raise ValueError("You should read a previous created adjacency matrix or provide a DataFrame.")

        self.logger.info("Creating Graph from Adjacency Matrix...")
        A = adj.values
        G = igraph.Graph.Adjacency((A > 0).tolist())
        G.es["weight"] = A[A.nonzero()]
        G.vs["label"] = adj.index

        self.logger.info("Graph created")
        g_n_vertices = G.vcount()
        g_n_edges = G.ecount()
        g_avg_degree = np.mean(G.degree())
        network_summary = pd.DataFrame.from_dict(
            {
                "Number of vertices (nodes)": [g_n_vertices],
                "Number of edges": [g_n_edges],
                "Average Degree": [g_avg_degree],
            }
        )
        self.logger.info("Network summary ready.")
        file_name = f"{self.path_prefix}__network_summary"
        save_and_log_table(
            df=network_summary, file_name=file_name, file_path=self.table_path, formats=["csv"], logger=self.logger
        )

        return G

    def detect_communities(
        self,
        adj: pd.DataFrame = None,
        read: bool = False,
        write: bool = True,
        quality_fn: str = "modularity",
        n_iterations: int = 10,
        max_comm_size: int = 120,
        seed: int = 42,
        **kwargs,
    ) -> dict:

        G = self._graph_from_matrix(adj, read=read)

        _quality = dict(
            modularity=leiden.ModularityVertexPartition,
            surprise=leiden.SurpriseVertexPartition,
            potts=leiden.CPMVertexPartition,
        )

        quality_method = _quality.get(quality_fn)

        self.logger.info(f"Running community detection algorithm using {quality_fn}")
        part = leiden.find_partition(
            G,
            partition_type=quality_method,
            weights="weight",
            n_iterations=n_iterations,
            seed=seed,
            max_comm_size=max_comm_size,
            **kwargs,
        )
        G.vs["community"] = part.membership

        p_quality = part.quality()
        p_modularity = part.modularity
        p_summary = part.summary()

        partitions_summary = pd.DataFrame.from_dict(
            {"Partitions Summary": [p_summary], "Quality": [p_quality], "Modularity": [p_modularity]}
        )

        self.logger.info("Partitions summary ready.")
        file_name = f"{self.path_prefix}__partition_summary"
        save_and_log_table(
            df=partitions_summary,
            file_name=file_name,
            file_path=self.table_path,
            formats=["csv", "tex"],
            logger=self.logger,
        )
        community_map = dict(zip(G.vs["label"], G.vs["community"]))
        if write:
            file_name = f"{self.path_prefix}__community_map"
            file = self.json_path / f"{file_name}.json"
            json.dump(community_map, open(file, "w"))

        return community_map

    def map_entities_to_communities(self, community_map: dict = None, df: pd.DataFrame = None, write=True, read=False):
        if read or not community_map:
            file_name = f"{self.path_prefix}__community_map"
            file = self.json_path / f"{file_name}.json"
            with open(file) as json_file:
                community_map = json.load(json_file)
        elif community_map is None:
            raise ValueError("You should read a previous created 'community_map' or provide a dictionary.")

        if df is None:
            entity_col = ENTITY_DICT.get(self.main_entity)
            df = load_donations_to_candidates(self.data_path, self.year, self.state, self.role, logger=self.logger)

        df = entity_subset(df, column=entity_col, entity=self.main_entity_subset)
        self.logger.info(f"Adding communities to 'donations to candidates' DataFrame...")
        df.loc[:, self.community_column_name] = df.loc[:, entity_col].map(community_map)
        if write:
            file_name = f"{self.path_prefix}_{self.main_entity}_{self.main_entity_subset}__communities"
            save_and_log_table(
                df=df, file_name=file_name, file_path=self.table_path, formats=["csv"], logger=self.logger
            )

        return df

    def _graph_to_matrix(self, G, adj: pd.DataFrame = None, sort_by: str = None, read=True, draw=True):
        if read:
            raise NotImplementedError
        elif adj is None:
            raise ValueError("You should read a previous created adjacency matrix or provide a DataFrame.")

        if G is None:
            raise ValueError("A Graph should be provided.")
            # check the possibility to save and load G igraph.Graph.load(self.data_path / f'{self.path_prefix}__network.net')
        community_dict = dict(zip(G.vs["label"], G.vs["community"]))
        comms = defaultdict(list)
        for x, y in community_dict.items():
            comms[y].append(x)
        comms_values = comms.values()
        if not sort_by:
            nodes_list = [node for comm in comms for node in comm]
        elif sort_by == "pr":
            nodes_list = sort_pagerank(G, comms_values=comms_values)
        else:
            NotImplemented(f'The method {sort_by} was not implemented. Only pagerank ("pr") is available.')

        A = adj.loc[nodes_list, nodes_list]
        A = np.asarray(A)

        if draw:
            self.logger.info(f"Drawing Adjacency Matrix figure...")
            draw_adjacency_matrix(
                A,
                comms_values,
                output_file=self.figure_path / "tex" / f"{self.path_prefix}__adj_matrix.pgf",
                logger=self.logger,
            )

        return A

    def benford_plot(self, df: pd.DataFrame = None, read=False, log_scale=False):
        if read:
            entity_col = ENTITY_DICT.get(self.main_entity)
            df = load_donations_to_candidates(self.data_path, self.year, self.state, self.role, logger=self.logger)
        elif df is None:
            raise ValueError("You should read a previous created 'donations to candidates' or provide a DataFrame.")

        df = entity_subset(df, column=entity_col, entity=self.main_entity_subset)
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

        plt.savefig(self.figure_path / "tex" / f"{self.path_prefix}_benford_distribution.pgf", bbox_inches="tight")
        self.logger.info(
            f"Benford Law figure saved: {self.figure_path}/tex/{self.path_prefix}__benford_distribution.pgf"
        )

    def benford_table(self, df=None, write=True, read=False):
        if read:
            raise NotImplementedError
        elif df is None:
            raise ValueError("You should read a previous created 'donations to candidates' or provide a DataFrame.")

        self.logger.info(f"Generating Benford Law table...")

        ## Benford Law Table
        benford_table = bf.benford_digits_table(df, self.community_column_name)

        benford_table = benford_table.loc[benford_table["N"] >= self.min_obs]

        if write:
            file_name = f"{self.path_prefix}__benford_law_{self.community_column_name}_gt_{self.min_obs}"
            save_and_log_table(
                df=benford_table, file_name=file_name, file_path=self.table_path, formats=["csv"], logger=self.logger
            )

        return benford_table

    def benford_dirty_stats_table(
        self, df=None, benford_table=None, write=True, read_df=False, read_benford_table=False, name="20190301"
    ):
        CNEP = pd.read_csv(self.data_path / f"{name}_CNEP.csv")
        CEIS = pd.read_csv(self.data_path / f"{name}_CEIS.csv")

        if read_df:
            raise NotImplementedError
        elif df is None:
            raise ValueError("You should read a previous created 'donations to candidates' or provide a DataFrame.")

        if read_benford_table:
            raise NotImplementedError
        elif benford_table is None:
            raise ValueError("You should read a previous created 'benford table' or provide a DataFrame.")

        cnpjs_dirty = pd.concat([CNEP["CPF OU CNPJ DO SANCIONADO"], CEIS["CPF OU CNPJ DO SANCIONADO"]]).apply(
            lambda x: str(x).zfill(14)
        )  # < contem cpfs tambem
        self.logger.info(f"Number of dirty companies: {cnpjs_dirty.nunique()}")

        df.loc[:, "id_donator_effective_cpf_cnpj_str"] = df.id_donator_effective_cpf_cnpj.apply(
            lambda x: str(int(x)).zfill(14)
        )

        df.loc[:, "dirty"] = df["id_donator_effective_cpf_cnpj_str"].isin(cnpjs_dirty)
        dirty_df = df.loc[df["dirty"]]
        self.logger.info(f"Number dirty donations: {dirty_df.shape[0]}")

        n_donnors_dirty = dirty_df.loc[:, "id_donator_effective_cpf_cnpj_str"].nunique()
        self.logger.info(f"Number of dirty donnors: {n_donnors_dirty}")

        # contagem de doacoes sujas nas community
        benford_table_dirty_donations = (
            df.groupby(self.community_column_name)
            .agg({"dirty": lambda x: int(sum(x))})
            .sort_values(by="dirty", ascending=False)
        )
        benford_table_dirty_donations.columns = ["# Dirty Donations"]
        benford_table_dirty_donations = benford_table.merge(
            benford_table_dirty_donations, how="left", on=self.community_column_name
        )

        # numero de donnors na comunidade
        donnors = df.groupby(self.community_column_name).agg(
            {"id_donator_effective_cpf_cnpj": lambda x: int(x.nunique())}
        )
        donnors.columns = ["# Donors"]
        benford_table_donnors = benford_table_dirty_donations.merge(donnors, how="left", on=self.community_column_name)

        # numero de doares dirty na comunidade
        donnors_dirty = dirty_df.groupby(self.community_column_name).agg(
            {"id_donator_effective_cpf_cnpj": lambda x: int(x.nunique())}
        )
        donnors_dirty.columns = ["# Dirty Donors"]
        benford_table_donnors_dirty = benford_table_donnors.merge(
            donnors_dirty, how="left", on=self.community_column_name
        )

        # total dirty doado na comunidade
        total_dirty_amount = dirty_df.groupby(self.community_column_name).agg(
            {"num_donation_ammount": lambda x: sum(x)}
        )
        total_dirty_amount.columns = ["Total Dirty Amount"]
        benford_table_total_dirty_amount = benford_table_donnors_dirty.merge(
            total_dirty_amount, how="left", on=self.community_column_name
        )

        # Soma das doacoes na comunidade:
        total_amount = df.groupby(self.community_column_name).agg({"num_donation_ammount": sum})
        total_amount.columns = ["Total Amount"]
        benford_table_total_amount = benford_table_total_dirty_amount.merge(
            total_amount, how="left", on=self.community_column_name
        )

        # Soma das doacoes na comunidade:
        donnors = df.groupby(self.community_column_name).agg({"id_candidate_cpf": lambda x: int(x.nunique())})
        donnors.columns = ["Number of Candidates"]
        benford_table_donnors = benford_table_total_amount.merge(
            donnors, how="left", on=self.community_column_name
        ).fillna(0)

        if write:
            file_name = f"{self.path_prefix}__benford_law_{self.community_column_name}_gt_{self.min_obs}_dirty_stats"
            save_and_log_table(
                df=benford_table_donnors,
                file_name=file_name,
                file_path=self.table_path,
                formats=["csv", "tex"],
                logger=self.logger,
            )

        return benford_table_donnors

    def generative_model(self, df: pd.DataFrame, should_fit=None, benford_table=None):
        should_fit = should_fit if should_fit is not None else self.should_fit
        self.selected_communities = benford_table[self.community_column_name].unique()

        selected_idx = df[self.community_column_name].isin(self.selected_communities)
        df = df.loc[selected_idx]

        name = f"{self.state}__generative_model_{self.community_column_name}_gt_{self.min_obs}"
        if should_fit == True:
            self.fit(df, group_column_name=self.community_column_name, name=name)
        self.write_latex_tables(name=name)
        self.compile_latex_tables()
        self.save_figures(group_list=self.selected_communities, group_column_name=self.community_column_name, name=name)
        # self.compile_latex_figures()
        parameters = pd.read_csv(self.table_path / "csv" / f"brazil_{self.year}_{self.role}_{name}__fit_parameters.csv")

        benford_table_fit_parameters = benford_table.join(parameters)

        file_name = f"{self.path_prefix}__benford_law_{self.community_column_name}_gt_{self.min_obs}_dirty_fit_params"

        save_and_log_table(
            df=benford_table_fit_parameters,
            file_name=file_name,
            file_path=self.table_path,
            formats=["csv", "tex"],
            logger=self.logger,
        )

        return benford_table_fit_parameters

    
