import logging
import os
from operator import itemgetter
from pathlib import Path

import pandas as pd


def save_and_log_table(
    df: pd.DataFrame, file_path: Path, file_name: str, formats: list = ["csv", "tex"], logger: logging.Logger = None
):
    for format in formats:
        file = file_path / format / f"{file_name}.format"
        if format == "csv":
            df.to_csv(file, index=False)
        elif format == "tex":
            df.style.to_latex(buf=file)
        else:
            raise ValueError("formats should be one or both 'csv' and/or 'tex'")
        logger.info(f"Saved in {format} format at: {file}")


def entity_subset(df: pd.DataFrame, column: str, entity: str):
    if entity == "companies":
        entity_condition = 14
    elif entity == "persons":
        entity_condition = 11
    else:
        raise ValueError("Entity should be one of 'companies' or 'persons'.")

    entity_idx = df[column].astype(str).apply(lambda x: len(x)) == entity_condition
    
    return df.loc[entity_idx]


def load_donations_to_candidates(data_path, year, state, role, logger: logging.Logger) -> pd.DataFrame:
    logger.info("Loading donations to candidates DataFrame from prepared data...")
    data_types = {
        "id_accountant_cnpj": str,
        "id_candidate_cpf": str,
        "id_donator_cpf_cnpj": str,
        "id_original_donator_cpf_cnpj": str,
        "id_donator_effective_cpf_cnpj": str,
    }

    donations_file = data_path / f"brazil_{year}_{role}_candidates_donations.csv"
    candidatos = pd.read_csv(
        donations_file,
        low_memory=False,
        dtype=data_types,
    )

    df = candidatos[
        (candidatos["id_donator_effective_cpf_cnpj"].isnull() == False) & (candidatos["cat_federative_unity"] == state)
    ]

    logger.info("Donations to candidates DataFrame loaded.")
    return df


def sort_pagerank(G, comms_values):
    pr = dict(zip(G.vs["label"], G.pagerank(weights="weight")))

    nodes_ordered = []
    for comm in comms_values:
        nodePR = []
        for node in comm:
            nodePR.append((node, pr[node]))
        nodePR = sorted(nodePR, key=itemgetter(1))  # sort list tuples(node, pr) from lower to higher pr
        nodes_ordered.append(nodePR)  # create a list of list of ordered tuples(node, pr) from lower to higher pr
        # This results in a list of communities ordered internally by PR
    nodes_list = []
    for sublist in nodes_ordered:
        for item in sublist:
            nodes_list.append(item[0])  # Same but only nodes

    return nodes_list


def create_logger(name: str = "Chrimatocracy", log_file: str = "log.log", level="DEBUG"):
    file_path = os.path.dirname(os.path.abspath(__file__))
    formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    consoleHandler = logging.StreamHandler()
    consoleHandler.setLevel(level)
    consoleHandler.setFormatter(formatter)

    fileHandler = logging.FileHandler(os.path.join(file_path, log_file))
    fileHandler.setLevel(level)
    fileHandler.setFormatter(formatter)

    logger = logging.getLogger()
    logger.setLevel(level)
    logger.addHandler(consoleHandler)
    logger.addHandler(fileHandler)

    return logger
