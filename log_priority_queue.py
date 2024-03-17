"""Main program flow for 'log_priority_queue'."""

import argparse
import logging
from datetime import datetime
from pathlib import Path
from typing import Final

import pandas as pd

from data import DataLoader
from model import OutlierModel

MIN_CLUSTER_SIZE: Final = 130
MIN_NO_CLUSTERS: Final = 1


def main() -> None:
    """Main function."""
    # argument parsing
    parser = argparse.ArgumentParser(
        prog="log_priority_queue",
        description=(
            "log_priority_queue is an application that sets a priority "
            "number to each log message in a batch of log messages by "
            "using rules, NLP data processing tools and an unsupervised "
            "outlier detection model."
        ),
        usage="""
        Example 1: input is an excel file:
              python log_priority_queue.py
                    --train_period_start 2023-03-17
                    --train_period_end 2023-04-17
                    --predict_period_start 2023-04-18
                    --predict_period_end 2023-04-18
                    excel
                    --file filename.xlsx

        Example 2: input is a database table:
              python log_priority_queue.py
                    --train_period_start 2023-03-17
                    --train_period_end 2023-04-17
                    --predict_period_start 2023-04-18
                    --predict_period_end 2023-04-18
                    database
                    --server_name SERVER-MSS123XYZ
                    --database_name DB_NAME
                    --schema schema_name
                    --table_name table_name
              """,
        epilog=("Application code is hosted on github"),
    )

    parser.add_argument(
        "--train_period_start",
        help=("The start of the training period in format 'YYYY-MM-DD'"),
        type=str,
    )

    parser.add_argument(
        "--train_period_end",
        help=("The end of the training period in format 'YYYY-MM-DD'"),
        type=str,
    )

    parser.add_argument(
        "--predict_period_start",
        help=("The start of the predict period in format 'YYYY-MM-DD'"),
        type=str,
    )

    parser.add_argument(
        "--predict_period_end",
        help=("The end of the predict period in format 'YYYY-MM-DD'"),
        type=str,
    )

    subparser = parser.add_subparsers(
        dest="subparser_name",
        help="Use either 'excel' or 'database'",
    )
    parser_excel = subparser.add_parser(
        "excel",
        help="pass an excel file for processing",
    )
    parser_database = subparser.add_parser(
        "database",
        help="pass a database table for processing",
    )
    parser_excel.add_argument(
        "--file",
        help="file name of excel file (excel)",
        type=str,
    )
    parser_database.add_argument(
        "--server_name",
        help="SQL server name (database)",
        type=str,
    )
    parser_database.add_argument(
        "--database_name",
        help="Database name (database)",
        type=str,
    )
    parser_database.add_argument(
        "--schema",
        help="Database schema (database)",
        type=str,
    )
    parser_database.add_argument(
        "--table_name",
        help="Database table name (database)",
        type=str,
    )
    args = parser.parse_args()

    # set runtime
    runtime = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    replace_map = str.maketrans(
        {
            " ": "_",
            "-": "_",
            ":": "_",
        },
    )
    runtime_name = runtime.translate(replace_map)

    # logging config
    logging.basicConfig(
        format=("%(asctime)s %(levelname)-8s [%(filename)s:%(lineno)d] %(message)s"),
        datefmt="%Y-%m-%d %H:%M:%S",
        level=logging.INFO,
        filename=Path("log") / f"logfile-{runtime_name}.txt",
    )
    logger = logging.getLogger(__name__)
    logger.info("*** Initiate logging ***")
    logger.info("Command-line arguments: %s", args)

    logger.info(
        "Minimum cluster size is %s and MIN_NO_CLUSTERS is %s",
        MIN_CLUSTER_SIZE,
        MIN_NO_CLUSTERS,
    )

    logger.info("*** Load data ***")

    # the argument parser catches unknown subparser names
    # - thus, object corpuses is not unbound
    if args.subparser_name == "database":
        logger.info("Use class method Dataloader.from_table")
        corpuses = DataLoader.from_dbtable(
            server_name=args.server_name,
            database_name=args.database_name,
            schema=args.schema,
            table_name=args.table_name,
            train_period_start=args.train_period_start,
            train_period_end=args.train_period_end,
            predict_period_start=args.predict_period_start,
            predict_period_end=args.predict_period_end,
        )
    elif args.subparser_name == "excel":
        logger.info("Use class method Dataloader.from_excel")
        corpuses = DataLoader.from_excel(
            path=args.file,
            train_period_start=args.train_period_start,
            train_period_end=args.train_period_end,
            predict_period_start=args.predict_period_start,
            predict_period_end=args.predict_period_end,
        )

    logger.info(
        "The data loader returned a train corpus containing %s log messages",
        len(corpuses.train_corpus),
    )
    logger.info(
        "Train corpus: %s",
        repr(corpuses.train_corpus),
    )
    logger.info(
        "The data loader returned a predict corpus containing %s log messages",
        len(corpuses.predict_corpus),
    )
    logger.info(
        "Predict corpus: %s",
        repr(corpuses.predict_corpus),
    )

    logger.info("*** Estimate outlier model ***")
    om = OutlierModel(corpuses.train_corpus, corpuses.predict_corpus, runtime)
    om.estimate(
        min_node_size=MIN_CLUSTER_SIZE,
        min_no_clusters=MIN_NO_CLUSTERS,
    )
    logger.info("The outlier model repr is: %s", repr(om))
    logger.info("*** Predict data ***")
    predictions = om.predict()

    logger.info("*** Write output ***")
    df_predict_delivery = pd.DataFrame(predictions)
    logger.info("Output data size is %s", repr(df_predict_delivery.shape))
    if args.predict_period_start == args.predict_period_end:
        output_str = (
            f"logdata_predict_{args.predict_period_start}"
            f"_runtime_{runtime_name}.xlsx"
        )
    else:
        output_str = (
            f"logdata_predict_{args.predict_period_start}_to_"
            f"{args.predict_period_end}_runtime_{runtime_name}.xlsx"
        )
    df_predict_delivery.to_excel(output_str, index=False)
    logger.info("*** End of execution ***")
    logging.shutdown()


if __name__ == "__main__":
    main()
