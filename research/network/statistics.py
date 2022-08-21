import warnings

warnings.simplefilter(action="ignore", category=FutureWarning)

import logging
import os
from pathlib import Path

from chrimatocracy.utils import load_donations, donations_made_stats, donations_received_stats

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

    fileHandler = logging.FileHandler(os.path.join(file_path, "statistics.log"))
    fileHandler.setLevel(logging.DEBUG)
    fileHandler.setFormatter(formatter)

    logger = logging.getLogger("Statistics")
    logger.setLevel(logging.DEBUG)
    logger.addHandler(consoleHandler)
    logger.addHandler(fileHandler)

    year = 2014

    data_path = Path(cwd) / "data" / "prepared" / f"{year}/"
    table_path = Path(file_dir) / "tables/tex/"
    figure_path = Path(file_dir) / "figures/"

    roles = ["state_deputy", "senator", "federal_deputy", "state_deputy", "district_deputy", "president"]

    directories = [table_path, figure_path]

    for directory in directories:
        if not os.path.exists(directory):
            os.makedirs(directory)

    for role in roles:
        logger.info(f"Generating Statistics table for {role}.\n")
        df = load_donations(role=role, year=year, data_path=data_path)
        stats_donations = donations_received_stats(df, year=2014, role=role, table_path=table_path)
        stats_companies = donations_made_stats(by='companies', df=df, year=2014, role=role, table_path=table_path)
        stats_sector    = donations_made_stats(by='sector', df=df, year=2014, role=role, table_path=table_path)

