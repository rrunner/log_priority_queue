"""Test DataLoader class."""

from datetime import date, timedelta

import pytest

from base import Corpus
from data import DataLoader


def test_incorrect_dates():
    """Test ValueError is raised on incorrect dates."""
    with pytest.raises(ValueError):
        DataLoader.from_dbtable(
            server_name="server_name",
            database_name="db_name",
            schema="schema_name",
            table_name="schema_name.table_name",
            train_period_start="2023-04-17",
            train_period_end="2023-04-17",
            predict_period_start="2023-04-18",
            predict_period_end="2023-04-18",
        )

    with pytest.raises(ValueError):
        DataLoader.from_excel(
            path="input/logdata.xlsx",
            train_period_start="2023-03-17",
            train_period_end="2023-04-17",
            predict_period_start="2023-04-17",
            predict_period_end="2023-04-18",
        )

    with pytest.raises(ValueError):
        one_day_ahead = date.today() + timedelta(days=1)
        one_day_ahead = one_day_ahead.strftime("%Y-%m-%d")
        DataLoader.from_excel(
            path="input/logdata.xlsx",
            train_period_start="2023-03-17",
            train_period_end="2023-04-17",
            predict_period_start="2023-04-18",
            predict_period_end=one_day_ahead,
        )


def test_excel_produces_corpus_data(create_excel_file):
    """Integration test: take excel input and verify that train and
    predict corpus are created."""
    corpuses = DataLoader.from_excel(
        path=create_excel_file,
        train_period_start="2023-04-01",
        train_period_end="2023-04-17",
        predict_period_start="2023-04-18",
        predict_period_end="2023-04-18",
    )
    assert isinstance(corpuses.train_corpus, Corpus), "the train corpus must be of type Corpus"
    assert isinstance(corpuses.predict_corpus, Corpus), "the predict corpus must be of type Corpus"
    assert len(corpuses.train_corpus) == 10, "the length of the train corpus must be 10 (status excluded)"
    assert len(corpuses.predict_corpus) == 12, "the length of the predict corpus must be 12"
