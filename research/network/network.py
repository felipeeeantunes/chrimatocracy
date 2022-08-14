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
    role = "state_deputy"  # [], "senator", "federal_deputy", "state_deputy", "district_deputy", "president"]
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

    

    # for role in roles:
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

    if use_previous_data:
        chrimnet.load_data()
        chrimnet.create_adj_matrix()
        chrimnet.detect_communities()
        chrimnet.draw_adj_matrix()
        chrimnet.benford_plot()
        chrimnet.benford_tex()
        chrimnet.generative_model(should_fit=False)
        chrimnet.gen_list()
    else:
        chrimnet.load_data()
        print(1)
        df = chrimnet.candidates_by_state
        chrimnet.create_adj_matrix(df=df)
        print(2)
        adj = chrimnet.adj
        chrimnet.detect_communities(adj=adj)
        print(3)
        G = chrimnet.G
        chrimnet.draw_adj_matrix(G=G, adj=adj)
        print(4)
        donations = df["num_donation_ammount"]
        chrimnet.benford_plot(donations=donations)
        print(5)
        df_with_communities = chrimnet.candidates_by_state
        chrimnet.benford_tex(df=df_with_communities)
        print(6)
        chrimnet.generative_model(should_fit=True)
        chrimnet.gen_list(df=df_with_communities)
        print(7)
