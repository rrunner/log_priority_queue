"""Test PredictionResult and OutlierModel class."""

from dataclasses import FrozenInstanceError
from datetime import date

import pytest
from scipy.sparse._csr import csr_matrix
from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import TfidfVectorizer

from base import Corpus
from model import OutlierModel, PredictionResult


def test_raise_mod_predictionresult():
    """Test exception is raised mutating a PredictionResult.
    This type is supposed to be immutable."""
    with pytest.raises(FrozenInstanceError):
        pred_res = PredictionResult(
            priority=100,
            runtime="2012-10-01",
            date_created="2012-10-01",
            msg_raw="This is a test string",
            explanation="Successful execution",
        )
        pred_res.priority = 900


def test_types_outlier_model(create_train_predict_corpus):
    """Test types of OutlierModel instance."""
    corpus_train, corpus_predict = create_train_predict_corpus
    om = OutlierModel(corpus_train, corpus_predict, "2023-04-02")
    assert isinstance(om._train_vectorizer, TfidfVectorizer), "attribute _train_vectorizer must be of type TfidfVectorizer"
    assert isinstance(om._train_kmeans_model, KMeans), "attribute _train_kmeans_model must be of type KMeans"
    assert isinstance(om._train_data, Corpus), "attribute _train_data must be of type Corpus"
    assert isinstance(om._train_data_text, csr_matrix), "attribute must be of type csr_matrix"
    assert isinstance(om._predict_data, Corpus), "attribute _predict_data must be of type Corpus"
    assert isinstance(om._predict_data_text, csr_matrix), "attribute must be of type csr_matrix"
    assert isinstance(om._runtime, str), "attribute _runtime must be of type str"


def test_priority_order(priority):
    """Test priority order.

    Test the highest priorities. If these tests succeed, the tests for lower priorites (such as 200) should
    also succeed.
    """
    pr = priority
    assert len(pr) == 5, "the number of predictions must equal 5"

    assert pr[0].priority == 900, "the priority must equal 900"
    assert pr[0].runtime == "2023-04-02 21:00:00", "the runtime must equal '2023-04-02 21:00:00'"
    assert pr[0].date_created == "2023-04-02", "date_created must equal '2023-04-02'"
    assert pr[0].msg_raw == "Some kind of outlier. This can be an issue.", "msg_raw must equal 'Some kind of outlier. This can be an issue.'"
    assert pr[0].explanation == f"Genuine outlier! Log message type unobserved during {date(2023, 3, 1)} - {date(2023, 4, 1)}", "the explanation text is incorrect"

    assert pr[1].priority == 850, "the priority must equal 850"
    assert pr[1].runtime == "2023-04-02 21:00:00", "the runtime must equal '2023-04-02 21:00:00'"
    assert pr[1].date_created == "2023-04-02", "date_created must equal '2023-04-02'"
    assert pr[1].msg_raw == "System 5 times. Testing!", "msg_raw must equal 'System 5 times. Testing!'"
    assert pr[1].explanation == f"Borderline outlier. Log message type observed 5 times during {date(2023, 3, 1)} - {date(2023, 4, 1)}", "the explanation text is incorrect"

    assert pr[2].priority == 840, "the priority must equal 840"
    assert pr[2].runtime == "2023-04-02 21:00:00", "the runtime must equal '2023-04-02 21:00:00'"
    assert pr[2].date_created == "2023-04-02", "date_created must equal '2023-04-02'"
    assert pr[2].msg_raw == "System 10 times. Testing, testing $$$", "msg_raw must equal 'System 10 times. Testing, testing $$$'"
    assert pr[2].explanation == f"Rare log message. Log message type observed 10 times during {date(2023, 3, 1)} - {date(2023, 4, 1)}", "the explanation text is incorrect"

    # this message occurs twice in predict data (hence index 3 and 4 below)
    assert pr[3].priority == 830, "the priority must equal 830"
    assert pr[3].runtime == "2023-04-02 21:00:00", "the runtime must equal '2023-04-02 21:00:00'"
    assert pr[3].date_created == "2023-04-02", "date_created must equal '2023-04-02'"
    assert pr[3].msg_raw == "hard psuedo text, 14", "msg_raw must equal 'hard psuedo text, 14'"
    assert pr[3].explanation == f"Rare log message. Log message type observed 14 times during {date(2023, 3, 1)} - {date(2023, 4, 1)}", "the explanation text is incorrect"
    assert pr[4].priority == 830, "the priority must equal 830"
    assert pr[4].runtime == "2023-04-02 21:00:00", "the runtime must equal '2023-04-02 21:00:00'"
    assert pr[4].date_created == "2023-04-02", "date_created must equal '2023-04-02'"
    assert pr[4].msg_raw == "hard psuedo text, 14", "msg_raw must equal 'hard psuedo text, 14'"
    assert pr[4].explanation == f"Rare log message. Log message type observed 14 times during {date(2023, 3, 1)} - {date(2023, 4, 1)}", "the explanation text is incorrect"


def test_success_failure(success_failure):
    """Test success and failure.

    Note that the test data is created in order 100 and 700, but the assert statements
    are in the reversed order because of descending sorting order of priority.
    """
    pr = success_failure
    assert len(pr) == 2, "the number of predictions must equal 2"

    # error
    assert pr[0].priority == 700, "the priority must equal 700"
    assert pr[0].runtime == "2023-04-02 21:00:00", "the runtime must equal '2023-04-02 21:00:00'"
    assert pr[0].date_created == "2023-04-02", "date_created must equal '2023-04-02'"
    assert pr[0].msg_raw == "System ZYX, this is an error", "msg_raw must equal 'System ZYX, this is an error'"
    assert pr[0].explanation == "Execution status 'Failure' (failed jobs are always reported with priority 700, independent of frequency)", "the explanation text is incorrect"

    # success
    assert pr[1].priority == 100, "the priority must equal 100"
    assert pr[1].runtime == "2023-04-02 21:00:00", "the runtime must equal '2023-04-02 21:00:00'"
    assert pr[1].date_created == "2023-04-02", "date_created must equal '2023-04-02'"
    assert pr[1].msg_raw == "System XYZ, this is a success", "msg_raw must equal 'System XYZ, this is a success'"
    assert pr[1].explanation == "Execution status 'Success' (successful jobs are always reported with priority 100)", "the explanation text is incorrect"
