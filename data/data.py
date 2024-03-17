"""Data related tools used to process log data."""

from __future__ import annotations

import logging
from datetime import date, datetime, time

import pandas as pd
import polars as pl
from sqlalchemy import VARCHAR, DateTime, Integer, and_, create_engine
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, sessionmaker

from base import Corpus, LogMsg

logger = logging.getLogger(__name__)


class DataLoader:
    """Data parser class.

    Load data locally in Excel format or parse from database table directly,
    as managed via class methods.

    Attributes
    ----------
        train_corpus: train data to be used by the application
        predict_corpus: predict data to be used by the application

    """

    def __init__(self, train_corpus: Corpus, predict_corpus: Corpus) -> None:
        """Init method for DataLoader.

        Args:
        ----
            train_corpus: train data to be used by the application
            predict_corpus: predict data to be used by the application

        """
        self.train_corpus = train_corpus
        self.predict_corpus = predict_corpus

    def __repr__(self) -> str:
        """Object representation.

        Returns
        -------
            A repr of the object.

        """
        return (
            f"<DataLoader(train_corpus={repr(self.train_corpus).strip('<>')}, "
            f"predict_corpus={repr(self.predict_corpus).strip('<>')})>"
        )

    @staticmethod
    def validate_dates(
        t_start: date,
        t_end: date,
        p_start: date,
        p_end: date,
    ) -> None:
        """Validate dates.

        Args:
        ----
            t_start: start of train period
            t_end: end of train period
            p_start: start of predict period
            p_end: end of predict period

        """
        if not t_start < t_end < p_start <= p_end <= date.today():
            logger.error("Dates in validate_dates are incorrect")
            logger.error("Train start: %s", t_start)
            logger.error("Train end: %s", t_end)
            logger.error("Predict start: %s", p_start)
            logger.error("Predict end: %s", p_end)
            logging.shutdown()
            raise ValueError(
                "Dates denoting train and predict periods are incorrect.",
            )

    @classmethod
    def from_excel(
        cls,
        path: str,
        train_period_start: str,
        train_period_end: str,
        predict_period_start: str,
        predict_period_end: str,
    ) -> DataLoader:
        """Class method to create DataLoader from local Excel file using
        polars.

        Args:
        ----
            path: path to Excel file
            train_period_start: start date of train period (fmt YYYY-MM-DD)
            train_period_end: end date of train period (fmt YYYY-MM-DD)
            predict_period_start: start date of predict period (fmt YYYY-MM-DD)
            predict_period_end: end date of predict period (fmt YYYY-MM-DD)

        Returns:
        -------
            a DataLoader instance

        """
        train_start = datetime.strptime(train_period_start, "%Y-%m-%d").date()
        train_end = datetime.strptime(train_period_end, "%Y-%m-%d").date()
        predict_start = datetime.strptime(
            predict_period_start,
            "%Y-%m-%d",
        ).date()
        predict_end = datetime.strptime(
            predict_period_end,
            "%Y-%m-%d",
        ).date()

        cls.validate_dates(
            train_start,
            train_end,
            predict_start,
            predict_end,
        )

        df_main = (
            pl.read_excel(path)
            .with_columns(
                pl.col("DateTimeCreated")
                .str.strptime(pl.Date, "%Y-%m-%d %H:%M:%S%.f")
                .alias("date_created"),
            )
            .filter(
                pl.col("date_created").is_between(
                    train_start,
                    predict_end,
                ),
            )
            .select(["date_created", "NotificationMessage"])
        )

        # create train corpus
        train_corpus = Corpus(train_start, train_end)
        df_train = df_main.filter(
            pl.col("date_created").is_between(
                train_start,
                train_end,
            ),
        )
        for row in df_train.iter_rows(named=True):
            log_msg = LogMsg(row["date_created"], row["NotificationMessage"])
            train_corpus.append(log_msg)

        # create predict corpus
        predict_corpus = Corpus(predict_start, predict_end)
        df_predict = df_main.filter(
            pl.col("date_created").is_between(predict_start, predict_end),
        )
        for row in df_predict.iter_rows(named=True):
            log_msg = LogMsg(row["date_created"], row["NotificationMessage"])
            predict_corpus.append(log_msg, predict=True)

        return cls(train_corpus, predict_corpus)

    @classmethod
    def from_dbtable(
        cls,
        server_name: str,
        database_name: str,
        schema: str,
        table_name: str,
        train_period_start: str,
        train_period_end: str,
        predict_period_start: str,
        predict_period_end: str,
    ) -> DataLoader:
        """Class method to create DataLoader from database table.

        Args:
        ----
            server_name: the server name
            database_name: the database name
            schema: the schema name
            table_name: name of database table
            train_period_start: start date of train period (fmt YYYY-MM-DD)
            train_period_end: end date of train period (fmt YYYY-MM-DD)
            predict_period_start: start date of predict period (fmt YYYY-MM-DD)
            predict_period_end: end date of predict period (fmt YYYY-MM-DD)

        Returns:
        -------
            an instance of DataLoader

        """
        train_start = datetime.strptime(train_period_start, "%Y-%m-%d").date()
        train_end = datetime.strptime(train_period_end, "%Y-%m-%d").date()
        predict_start = datetime.strptime(
            predict_period_start,
            "%Y-%m-%d",
        ).date()
        predict_end = datetime.strptime(
            predict_period_end,
            "%Y-%m-%d",
        ).date()

        cls.validate_dates(
            train_start,
            train_end,
            predict_start,
            predict_end,
        )

        # setup database connection
        conn_str = f"mssql+pyodbc://{server_name}/{database_name}?driver=SQL+Server"
        engine = create_engine(conn_str)
        session_maker = sessionmaker(bind=engine)

        class Base(DeclarativeBase):
            """Base class."""

        class LogTable(Base):
            """Python object representation of the database table.

            The table has no primary key but sqlalchemy requires one to be
            defined. Use DateTimeCreated as the primary key for this reason.
            """

            __tablename__ = table_name
            __table_args__ = {"schema": schema}
            MailTargetKey: Mapped[int] = mapped_column(Integer)
            ReturnedSeverityLevel: Mapped[int] = mapped_column(Integer)
            MailAddress: Mapped[str] = mapped_column(VARCHAR)
            DateTimeCreated: Mapped[DateTime] = mapped_column(
                DateTime,
                primary_key=True,
            )
            DateTimeSent: Mapped[DateTime] = mapped_column(DateTime)
            CreateMetaProcSubKey: Mapped[int] = mapped_column(Integer)
            SendMetaProcSubKey: Mapped[int] = mapped_column(Integer)
            NotificationMessage: Mapped[int] = mapped_column(VARCHAR)

            def __repr__(self) -> str:
                return (
                    f"<LogTable({self.DateTimeCreated!r}, "
                    f"{self.NotificationMessage!r}, ...)>"
                )

        # extract data from database
        with session_maker() as session:
            qry_result = (
                session.query(
                    LogTable.DateTimeCreated,
                    LogTable.NotificationMessage,
                )
                .filter(
                    and_(
                        LogTable.DateTimeCreated
                        >= datetime.combine(
                            train_start,
                            time(0, 0, 0),
                        ),
                        LogTable.DateTimeCreated
                        <= datetime.combine(
                            predict_end,
                            time(23, 59, 59, 999999),
                        ),
                    ),
                )
                .all()
            )
            df_main = pd.DataFrame(qry_result)

        df_train = df_main.loc[
            (
                train_start
                <= pd.to_datetime(
                    df_main["DateTimeCreated"],
                ).dt.date
            )
            & (
                pd.to_datetime(
                    df_main["DateTimeCreated"],
                ).dt.date
                <= train_end
            ),
            :,
        ]

        # train corpus
        train_corpus = Corpus(train_start, train_end)

        # parse training data, create log messages and add to corpus
        for _, row in df_train.iterrows():
            message_text = row.NotificationMessage
            date_created = pd.to_datetime(row.DateTimeCreated).date()
            msg = LogMsg(date_created, message_text)
            train_corpus.append(msg)

        # subset predict data
        df_predict = df_main.loc[
            (
                predict_start
                <= pd.to_datetime(
                    df_main["DateTimeCreated"],
                ).dt.date
            )
            & (
                pd.to_datetime(
                    df_main["DateTimeCreated"],
                ).dt.date
                <= predict_end
            ),
            :,
        ]

        # predict corpus
        predict_corpus = Corpus(predict_start, predict_end)

        # populate the corpus for the predictions
        for _, row in df_predict.iterrows():
            message_text = row.NotificationMessage
            date_created = pd.to_datetime(row.DateTimeCreated).date()
            msg = LogMsg(date_created, message_text)
            predict_corpus.append(msg, predict=True)

        return cls(train_corpus, predict_corpus)
