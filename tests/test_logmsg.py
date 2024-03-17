"""Test LogMsg class."""

from datetime import date
from typing import Tuple

from spacy.tokens.doc import Doc

from base import LogMsg


def test_types_success(logmsg_normal):
    """Test types of LogMsg instance."""
    assert isinstance(logmsg_normal, LogMsg), "class must be LogMsg"
    assert isinstance(logmsg_normal.date_created, date), "instance variable date_created must be date"
    assert isinstance(logmsg_normal.msg_raw, str), "instance variable msg_raw must be str"
    assert isinstance(logmsg_normal._doc, Doc), "instance variable _doc must be Doc"
    assert isinstance(logmsg_normal.lemmas, Tuple), "instance variable lemmas must be Tuple"
    assert isinstance(logmsg_normal.lemmas_text, str), "instance variable lemmas_text must be str"


def test_clean_success(logmsg_normal):
    """Test data cleansing of data."""
    lemmas = logmsg_normal.lemmas

    for lemma in lemmas:
        assert "<TD>" not in lemma, "<TD> must not be a lemma"
        assert "</TD>" not in lemma, "</TD> must not be a lemma"
        assert "(12 of 12)" not in lemma, "(12 of 12) must not be a lemma"
        assert "2023-02-13 04:00" not in lemma, "2023-02-13 04:00 must not be a lemma"


def test_message_date_created(logmsg_normal):
    """Test date_created."""
    assert logmsg_normal.date_created == date(2023, 2, 13)


def test_lemmas_normal(logmsg_normal):
    """Test lemmas."""
    exp_result = ('system', 'x', 'publish', 'time', 'dimension')
    assert logmsg_normal.lemmas == exp_result, "lemmas are not correct"


def test_lemmas_text_normal(logmsg_normal):
    """Test lemmas_text."""
    assert logmsg_normal.lemmas_text == "system x publish time dimension", "lemmas_text is not correct"


def test_length_normal(logmsg_normal):
    """Test length of 'normal'."""
    assert len(logmsg_normal) == 5, "length must be 5"


def test_success(logmsg_success):
    """Test success method."""
    status = logmsg_success.success()
    assert status, "success method must return True"


def test_lemmas_success(logmsg_success):
    """Test lemmas for 'success'."""
    assert logmsg_success.lemmas == ('gem', 'load', 'complete', 'publish', 'time', 'dimension'), "lemmas are not correct"


def test_lemmas_text_success(logmsg_success):
    """Test lemmas_text."""
    assert logmsg_success.lemmas_text == "gem load complete publish time dimension", "lemmas_text is not correct"


def test_length_success(logmsg_success):
    """Test length of success."""
    assert len(logmsg_success) == 6, "length must be 6"


def test_failure(logmsg_failure):
    """Test failure method."""
    status = logmsg_failure.failure()
    assert status, "failure method must return True"


def test_lemmas_failure(logmsg_failure):
    """Test lemmas for 'failure'."""
    assert logmsg_failure.lemmas == ('hrservice', 'freeze', 'check', 'date'), "lemmas are not correct"


def test_lemmas_text_failure(logmsg_failure):
    """Test lemmas_text."""
    assert logmsg_failure.lemmas_text == "hrservice freeze check date", "lemmas_text is not correct"


def test_length_failure(logmsg_failure):
    """Test length of failure."""
    assert len(logmsg_failure) == 4, "length must be 4"


def test_fail_texts(fail_texts):
    """Test failure texts."""
    assert fail_texts.failure(), "failure method must return True"
    assert len(fail_texts) == 0, "length must be 0"
    assert fail_texts.lemmas_text == "", "lemmas_text must be an empty string"


def test_success_texts(success_texts):
    """Test success texts."""
    assert success_texts.success(), "success method must return True"
    assert len(success_texts) == 0, "length must be 0"
    assert success_texts.lemmas_text == "", "lemmas_text must be an empty string"
