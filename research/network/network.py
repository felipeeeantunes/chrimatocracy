import warnings

warnings.simplefilter(action="ignore", category=FutureWarning)

import logging
import os
from pathlib import Path

from chrimatocracy.network import Network

cwd = Path().absolute()
file_dir = Path(__file__).parent.resolve()

if __name__ == "__main__":
    file_path = os.path.dirname(os.path.abspath(__file__))
    # create formatter
    formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")

    # add formatter to ch
    consoleHandler = logging.StreamHandler()
    consoleHandler.setLevel(logging.DEBUG)
    consoleHandler.setFormatter(formatter)

    fileHandler = logging.FileHandler(os.path.join(file_path, "network.log"))
    fileHandler.setLevel(logging.DEBUG)
    fileHandler.setFormatter(formatter)

    logger = logging.getLogger("Chrimatocracy")
    logger.setLevel(logging.DEBUG)
    logger.addHandler(consoleHandler)
    logger.addHandler(fileHandler)

    year = 2014
    roles = ["state_deputy"]  # [], "senator", "federal_deputy", "state_deputy", "district_deputy", "president"]
    state = "SP"
    community_column_name = "lv_community"
    use_previous_data = False
    min_obs = 100
    fit = True

    data_path = Path(cwd) / "data" / "prepared" / f"{year}/"
    table_path = Path(file_dir) / "tables/"
    figure_path = Path(file_dir) / "figures/"

    logger.info(
        f"""Chrimatocracy Network:\n
                 year={year}\n
                 data_path={data_path}\n
                 table_path={table_path}\n
                 figure_path={figure_path}\n
                 """
    )

    directories = [table_path, figure_path]

    for directory in directories:
        if not os.path.exists(directory):
            os.makedirs(directory)

    for role in roles:
        logger.info(f"Generating Network Analysis table for {role}.\n")
        chrimnet = Network(
            year=year,
            role=role,
            state=state,
            data_path=data_path,
            table_path=table_path,
            figure_path=figure_path,
            community_column_name=community_column_name,
            min_obs=min_obs,
            should_fit=fit,
            benford_log_scale=False,
            logger=logger,
        )

        df = chrimnet.load_donations_to_candidates()
        adj_df = chrimnet.create_adj_matrix(df=df, write=True)
        communities_g = chrimnet.detect_communities(adj=adj_df, read=False, write=True)
        communities_df = chrimnet.append_communities(df=df, G=communities_g, write=True)
        communities_adj, nodes_list, comms, colors = chrimnet.graph_to_matrix(
            G=communities_g, adj=adj_df, read=False, draw=True
        )
        benford_table_df = chrimnet.benford_table(df=communities_df, write=True, read=False)
        benford_dirty_stats_table_df = chrimnet.benford_dirty_stats_table(
            df=communities_df,
            benford_table=benford_table_df,
            write=True,
            read_df=False,
            read_benford_table=False,
            name="20190301",
        )
        benford_table_fit_parameters_df = chrimnet.generative_model(
            df=communities_df, should_fit=True, benford_table=benford_dirty_stats_table_df
        )
        ##Plots
        chrimnet.benford_plot(df=communities_df, read=False, log_scale=False)
