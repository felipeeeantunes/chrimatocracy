import warnings

warnings.simplefilter(action="ignore", category=FutureWarning)

import os
from pathlib import Path
import itertools

from chrimatocracy.network import Network
from chrimatocracy.utils import create_logger, load_donations_to_candidates

cwd = Path().absolute()
file_dir = Path(__file__).parent.resolve()

def run(
    year=2014,
    state='SP',
    role='state_deputy',
    quality_fn='modularity',
    max_size=100,
    n_iter=2,
    main_entity="donnors",
    relationship_entity="candidates",
    main_entity_subset="companies",
    min_obs=100,
    fit=True,
    logger=None,
):
   
        data_path = Path(cwd) / "data" / "prepared" / f"{year}/"
        table_path = Path(file_dir) / f"tables/{state}/{role}/{quality_fn}_{max_size}_{n_iter}/"
        table_path = Path(file_dir) / f"json/{state}/{role}/{quality_fn}_{max_size}_{n_iter}/"
        figure_path = Path(file_dir) / f"figures/{state}/{role}/{quality_fn}_{max_size}_{n_iter}/"

        logger.info(
            f"""Chrimatocracy Network:\n
                    year={year}\n
                    data_path={data_path}\n
                    table_path={table_path}\n
                    json_path={json_path}\n
                    figure_path={figure_path}\n
                    """
        )

        directories = [table_path, figure_path, json_path]

        for directory in directories:
            if not os.path.exists(directory):
                os.makedirs(directory)

        df = load_donations_to_candidates(data_path=data_path, year=year, state=state, role=role, logger=logger)

        logger.info(f"Generating Network Analysis table for {role}.\n")
        chrimnet = Network(
            year=year,
            role=role,
            state=state,
            data_path=data_path,
            table_path=table_path,
            figure_path=figure_path,
            main_entity=main_entity,
            relationship_entity=relationship_entity,
            main_entity_subset=main_entity_subset,
            min_obs=min_obs,
            should_fit=fit,
            benford_log_scale=False,
            logger=logger,
        )
        adj_df = chrimnet.projected_matrix(df=df, write=True)
        communities_map = chrimnet.detect_communities(
            adj=adj_df,
            read=False,
            write=True,
            quality_fn=quality_fn,
            n_iterations=n_iter,
            max_comm_size=max_size,
        )
        communities_df = chrimnet.map_entities_to_communities(df=df, community_map=communities_map, write=True)
        benford_table_df = chrimnet.benford_table(df=communities_df, write=True, read=False)
        benford_dirty_stats_table_df = chrimnet.benford_dirty_stats_table(
            df=communities_df,
            benford_table=benford_table_df,
            write=True,
            read_df=False,
            read_benford_table=False,
            name="20220826",
        )
        benford_table_fit_parameters_df = chrimnet.generative_model(
            df=communities_df, should_fit=True, benford_table=benford_dirty_stats_table_df
        )
        chrimnet.benford_plot(df=communities_df, read=False, log_scale=False)

if __name__ == "__main__":  

    logger = create_logger(name="Network", log_file="network.log", level="DEBUG")

    roles=["state_deputy"]
    states=["SP"]
    qualities=["modularity"]
    n_iters=[2]
    max_sizes=[0]
    years = [2014]

    iters = years, states, roles, qualities, n_iters, max_sizes

    for year, state, role, quality_fn, n_iter, max_size in itertools.product(*iters):
        run(
            year=year,
            main_entity="donnors",
            relationship_entity="candidates",
            main_entity_subset="companies",
            min_obs=100,
            fit=True,
            logger=logger,
        )


