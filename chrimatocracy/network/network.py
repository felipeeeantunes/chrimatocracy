import itertools
import logging
from collections import defaultdict

import igraph
import leidenalg as louvain
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import ticker

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
        year,
        role,
        state,
        data_path,
        table_path,
        figure_path,
        community_column_name,
        min_obs,
        should_fit=True,
        benford_log_scale=True,
        logger: logging.Logger = None,
    ):

        super().__init__(year, role, data_path, table_path, figure_path, community_column_name, logger)

        self.data_types = {
            "id_accountant_cnpj": str,
            "id_candidate_cpf": str,
            "id_donator_cpf_cnpj": str,
            "id_original_donator_cpf_cnpj": str,
            "id_donator_effective_cpf_cnpj": str,
        }

        self.should_fit = should_fit
        self.benford_log_scale = benford_log_scale
        self.state = state
        self.min_obs = min_obs
        self.logger = logger

    def load_data(self):
        candidatos = pd.read_csv(
            self.data_path / f"brazil_{self.year}_{self.role}_candidates_donations.csv",
            low_memory=False,
            dtype=self.data_types,
        )

        self.candidates_by_state = candidatos[
            (candidatos["id_donator_effective_cpf_cnpj"].isnull() == False)
            & (candidatos["cat_federative_unity"] == self.state)
        ]

    def create_adj_matrix(self, df=None):

        df = df if df is not None else self.candidates_by_state

        idx = df.id_candidate_cpf.unique()
        adj = pd.DataFrame(0, index=idx, columns=idx)

        for donator in idx:
            rec = df.loc[df.id_donator_effective_cpf_cnpj == donator, "id_candidate_cpf"].unique()
            cpf_pairs = list(itertools.combinations(rec, 2))
            for cpf1, cpf2 in cpf_pairs:
                adj.loc[cpf1, cpf2] += 1.0

        adj.reset_index(inplace=True)
        adj.to_csv(self.table_path / f"brazil_{self.year}_{self.role}_{self.state}__adj_matrix.csv", index=False)

        self.adj = adj.set_index("index")

    def detect_communities(self, adj=None):

        adj = (
            adj
            if adj is not None
            else pd.read_csv(
                self.table_path / f"brazil_{self.year}_{self.role}_{self.state}__adj_matrix.csv",
                converters={"index": lambda x: str(x).zfill(11)},
            ).set_index("index")
        )

        # Get the values as np.array, it's more convenenient.
        A = adj.values
        # Create graph, A.astype(bool).tolist() or (A / A).tolist() can also be used.
        G = igraph.Graph.Adjacency((A > 0).tolist())
        # Add edge weights and node labels.
        G.es["weight"] = A[A.nonzero()]
        G.vs["label"] = adj.index  # or a.index/a.columns

        part = louvain.find_partition(G, louvain.ModularityVertexPartition, weights="weight")
        p_summary = part.summary()
        p_quality = part.quality()
        p_modularity = part.modularity
        g_n_vertices = G.vcount()
        g_n_edges = G.ecount()
        g_avg_degree = np.mean(G.degree())

        G.vs["community"] = part.membership

        #%%
        # Simple check
        df_from_g = pd.DataFrame(G.get_adjacency(attribute="weight").data, columns=G.vs["label"], index=G.vs["label"])
        assert (df_from_g == adj).all().all()
        #%%
        self.logger.info(f"Partitions Summary: {p_summary}")
        self.logger.info(f"Quality:  {p_quality}")
        self.logger.info(f"Modularity:  {p_modularity}")
        self.logger.info(f"Number of vertices (nodes): {g_n_vertices}")
        self.logger.info(f"Number of edges: {g_n_edges}")
        self.logger.info(f"Average Degree: {g_avg_degree}")

        partitions_summary = pd.DataFrame.from_dict(
            {"Partitions Summary": [p_summary], "Quality": [p_quality], "Modularity": [p_modularity]}
        )
        partitions_summary.to_csv(
            self.table_path / f"brazil_{self.year}_{self.role}_{self.state}__partition_summary.csv"
        )
        with open(self.table_path / f"brazil_{self.year}_{self.role}_{self.state}_partitions_summary.tex", "w") as tf:
            tf.write(partitions_summary.to_latex(index=False))

        network_summary = pd.DataFrame.from_dict(
            {
                "Number of vertices (nodes)": [g_n_vertices],
                "Number of edges": [g_n_edges],
                "Average Degree": [g_avg_degree],
            }
        )
        network_summary.to_csv(self.table_path / f"brazil_{self.year}_{self.role}_{self.state}__network_summary.csv")
        with open(self.table_path / f"brazil_{self.year}_{self.role}_{self.state}_network_summary.tex", "w") as tf:
            tf.write(network_summary.to_latex(index=False))

        louvain_community_dict = dict(zip(G.vs["label"], G.vs["community"]))
        self.candidates_by_state.loc[:, "lv_community"] = self.candidates_by_state.loc[:, "id_candidate_cpf"].map(
            louvain_community_dict
        )
        self.candidates_by_state.to_csv(
            self.table_path / f"brazil_{self.year}_{self.role}_{self.state}__communities.csv", index=False
        )
        self.G = G

    def draw_adj_matrix(self, G=None, adj=None):
        #%%

        G = (
            G if G is not None else self.G
        )  # igraph.Graph.load(self.data_path / f'brazil_{self.year}_{self.role}_{self.state}__network.net')
        adj = (
            adj
            if adj is not None
            else pd.read_csv(
                self.data_path / f"brazil_{self.year}_{self.role}_{self.state}__adj_matrix.csv",
                converters={"index": lambda x: str(x).zfill(11)},
            ).set_index("index")
        )

        pr = dict(zip(G.vs["label"], G.pagerank(weights="weight")))
        maxPR = max(pr.values())

        flip_PR = dict(
            zip(
                sorted(dict(zip(G.vs["label"], G.pagerank(weights="weight"))), key=lambda node: pr[node], reverse=True),
                range(0, G.vcount()),
            )
        )

        pr_seq = sorted(pr, reverse=True)  # degree sequence

        #%%
        ## VERIFICAR A LEI DE BENFORD PARA CANDIDATOS A ESQUERDA DO A DIREITA

        #%%
        # Run louvain community finding algorithm
        louvain_community_dict = dict(zip(G.vs["label"], G.vs["community"]))

        # Convert community assignmet dict into list of communities
        louvain_comms = defaultdict(list)
        for node_index, comm_id in louvain_community_dict.items():
            louvain_comms[comm_id].append(node_index)
        louvain_comms = louvain_comms.values()

        nodes_louvain_ordered = [node for comm in louvain_comms for node in comm]

        #%%
        from operator import itemgetter

        nodes_ordered = []
        for comm in louvain_comms:
            nodePR = []
            for node in comm:
                nodePR.append((node, pr[node]))

            nodePR = sorted(nodePR, key=itemgetter(1))
            nodes_ordered.append(nodePR)

        #%%
        nodes_list = []
        for sublist in nodes_ordered:
            for item in sublist:
                nodes_list.append(item[0])

        #%%
        from random import randint

        cores = []

        for i in range(len(louvain_comms)):
            cores.append("%06X" % randint(0, 0xFFFFFF))
            # cores.append('blue')

        #%%

        draw_adjacency_matrix(
            adj,
            nodes_list,
            louvain_comms,
            colors=cores,
            output_file=self.figure_path / f"brazil_{self.year}_{self.role}_{self.state}__adj_matrix.pgf",
        )

    def benford_plot(self, donations=None):

        donations = donations if donations is not None else self.candidates_by_state["num_donation_ammount"]

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

        plt.savefig(self.figure_path / f"brazil_{self.year}_{self.role}_{self.state}_chrimnet_benford_distribution.pgf")

    def benford_tex(self, df=None, group=None, name="benford_law"):
        # name = benford_law

        df = (
            df
            if df is not None
            else pd.read_csv(self.data_path / f"brazil_{self.year}_{self.role}_{self.state}__communities.csv")
        )
        group = group if group is not None else self.community_column_name

        ## Benford Law Table
        benford_table = bf.benford_digits_table(df, self.community_column_name)
        name = name if self.state is None else f"{name}_{self.state}"
        name = name if self.min_obs is None else f"{name}__gt_{self.min_obs}"

        benford_table = benford_table.loc[benford_table["N"] > self.min_obs]
        benford_table.to_csv(
            self.table_path / f"brazil_{self.year}_{self.role}_{self.community_column_name}_{name}.csv", index=False
        )

        self.logger.info(f"Benford Law Results expected:\n benford_table")
        with open(
            self.table_path / f"brazil_{self.year}_{self.role}_{self.community_column_name}_{name}.tex", "w"
        ) as tf:
            tf.write(benford_table.to_latex(index=False))

        self.selected_communities = benford_table[self.community_column_name].unique()
        self.benford_table = benford_table

        return benford_table

    def gen_list(self, df=None, benford_table=None, name="benford_law"):
        CNEP = pd.read_csv(self.data_path / "20190301_CNEP.csv")
        CEIS = pd.read_csv(self.data_path / "20190301_CEIS.csv")

        name = name if self.state is None else f"{name}_{self.state}"
        name = name if self.min_obs is None else f"{name}__gt_{self.min_obs}"

        candidates_by_state = (
            df
            if df is not None
            else pd.read_csv(self.data_path / f"brazil_{self.year}_{self.role}_{self.state}__communities.csv")
        )
        benford_table = (
            benford_table
            if benford_table is not None
            else pd.read_csv(self.data_path / f"brazil_{self.year}_{self.role}_{self.community_column_name}_{name}.csv")
        )

        group = self.community_column_name

        #%%
        cnpjs_sujos = (
            pd.concat([CNEP["CPF OU CNPJ DO SANCIONADO"], CEIS["CPF OU CNPJ DO SANCIONADO"]])
            .apply(lambda x: str(x).zfill(14))
        )  # < contem cpfs tambem

        #%%
        self.logger.info(f"Numero de cnpjs sujos: {cnpjs_sujos.nunique()}")

        #%%
        candidates_by_state.loc[
            :, "id_donator_effective_cpf_cnpj_str"
        ] = candidates_by_state.id_donator_effective_cpf_cnpj.apply(lambda x: str(int(x)).zfill(14))

        #%%
        candidates_by_state.loc[:, "dirty"] = candidates_by_state["id_donator_effective_cpf_cnpj_str"].isin(cnpjs_sujos)

        #%%
        doadores_sujos = candidates_by_state.loc[
            candidates_by_state["dirty"], "id_donator_effective_cpf_cnpj_str"
        ].unique()
        n_doadores_sujos = candidates_by_state.loc[
            candidates_by_state["dirty"], "id_donator_effective_cpf_cnpj_str"
        ].nunique()

        #%%
        self.logger.info(f'Numero de doacoes de cpfs/cnpjs dirty: {candidates_by_state["dirty"].sum()}')

        #%%
        self.logger.info(f"Numero de cpfs/cnpjs dirty que doaram: {n_doadores_sujos}")

        #%%
        # contagem de doacoes sujas nas comunidades
        comunidades_sujos = (
            candidates_by_state.groupby(group).agg({"dirty": lambda x: sum(x)}).sort_values(by="dirty", ascending=False)
        )
        comunidades_sujos.columns = ["# Dirty Donations"]
        benford_table = benford_table.merge(comunidades_sujos, how="left", on=group)
        # benford_table = benford_table.join(comunidades_sujos)

        #%%
        # numero de doadores na comunidade
        comunidades_doadores = candidates_by_state.groupby(group).agg(
            {"id_donator_effective_cpf_cnpj": lambda x: x.nunique()}
        )
        comunidades_doadores.columns = ["# Donors"]
        benford_table = benford_table.merge(comunidades_doadores, how="left", on=group)

        #%%
        # numero de doares sujos na comunidade
        comunidades_doadores_sujos = (
            candidates_by_state.loc[candidates_by_state.id_donator_effective_cpf_cnpj.isin(doadores_sujos)]
            .groupby(group)
            .agg({"id_donator_effective_cpf_cnpj": lambda x: int(x.nunique())})
        )
        comunidades_doadores_sujos.columns = ["# Dirty Donors"]
        benford_table = benford_table.merge(comunidades_doadores_sujos, how="left", on=group)

        #%%
        # total sujo doado na comunidade
        comunidades_total_sujo = (
            candidates_by_state.loc[candidates_by_state.id_donator_effective_cpf_cnpj.isin(doadores_sujos)]
            .groupby(group)
            .agg({"num_donation_ammount": lambda x: sum(x)})
        )
        comunidades_total_sujo.columns = ["Total Dirty Amount"]
        benford_table = benford_table.merge(comunidades_total_sujo, how="left", on=group)

        #%%
        # Soma das doacoes na comunidade:
        comunidades_total_doacoes = candidates_by_state.groupby(group).agg({"num_donation_ammount": sum})
        comunidades_total_doacoes.columns = ["Total Amount"]
        benford_table = benford_table.merge(comunidades_total_doacoes, how="left", on=group)
        #%%
        # Soma das doacoes na comunidade:
        comunidades_numero_candidatos = candidates_by_state.groupby(group).agg({"num_donation_ammount": "count"})
        comunidades_numero_candidatos.columns = ["Number of Candidates"]
        benford_table = benford_table.merge(comunidades_numero_candidatos, how="left", on=group)

        parameters = pd.read_csv(
            self.data_path / f"brazil_{self.year}_{self.role}_{self.community_column_name}_{name}__parameters.csv"
        )
        #%%
        # Parametros da comunidade:
        benford_table = benford_table.join(parameters)

        #%%

        self.logger.info(f"Benford Law Results with dirty measure:\n {benford_table}")
        benford_table.to_csv(
            self.table_path / f"brazil_{self.year}_{self.role}_{name}__dirty_donations.csv", index=False
        )
        with open(self.table_path / f"brazil_{self.year}_{self.role}_{name}__dirty_donations.tex", "w") as tf:
            tf.write(
                benford_table.reset_index()
                .rename(columns={"index": "Community"})
                .to_latex(index=False, escape=False, bold_rows=True, float_format="{:0.2f}".format)
            )

    def generative_model(self, should_fit=None):
        should_fit = should_fit if should_fit is not None else self.should_fit
        selected_idx = self.candidates_by_state[self.community_column_name].isin(self.selected_communities)
        df = self.candidates_by_state.loc[selected_idx]
        name = "benford_law"
        name = name if self.state is None else f"{name}_{self.state}"
        name = name if self.min_obs is None else f"{name}__gt_{self.min_obs}"
        if should_fit == True:
            self.fit(df, name)

        self.write_latex_tables(name=name)
        self.save_figures(group_list=self.selected_communities, name=name)
