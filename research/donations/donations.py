import logging
import os
from pathlib import Path

from chrimatocracy.network import Donations

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

    fileHandler = logging.FileHandler(os.path.join(file_path, "donations.log"))
    fileHandler.setLevel(logging.DEBUG)
    fileHandler.setFormatter(formatter)

    logger = logging.getLogger("Chrimatocracy")
    logger.setLevel(logging.DEBUG)
    logger.addHandler(consoleHandler)
    logger.addHandler(fileHandler)

    year = 2014
    data_path = Path(cwd) / "data" / "prepared" / f"{year}/"
    table_path = Path(file_dir) / "tables/"
    figure_path = Path(file_dir) / "figures/"

    logger.info(
        f"""Chrimatocracy Analysis:\n
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

    # [], "senator", "federal_deputy", "state_deputy", "district_deputy", "president"]
    roles = ["federal_deputy"]
    # roles = ["president"]

    for role in roles:
        print(f"Generating Chrimatocracy Analysis table for {role}.\n")
        chrimatocracy = Donations(
            role=role,
            year=year,
            data_path=data_path,
            table_path=table_path,
            figure_path=figure_path,
            group="cat_federative_unity",
            should_fit=True,
            benford_log_scale=False,
            logger=logger,
        )

        chrimatocracy.load_donations()
        chrimatocracy.benford_plot()
        # chrimatocracy.lr_table()
        chrimatocracy.generative_model()
