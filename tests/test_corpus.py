"""Test Corpus class."""

from datetime import date

from base import Corpus, LogMsg


def test_types_corpus():
    """Test class name and types."""
    corpus = Corpus(date(2020, 1, 20), date(2020, 1, 31))
    assert isinstance(corpus, Corpus), "class must be Corpus"
    assert isinstance(corpus._corpus, list), "instance variable _corpus must be a list"
    assert isinstance(corpus.messages, list), "instance variable messages must be a list"
    assert isinstance(corpus.fromdate, date), "instance variable fromdate must be a date"
    assert isinstance(corpus.enddate, date), "instance variable enddate must be a date"
    assert isinstance(corpus.distinct_words, set), "instance variable distinct_words must be a set"


def test_empty_corpus() -> None:
    """Test length of empty corpus."""
    corpus = Corpus(date(2020, 1, 20), date(2020, 1, 31))
    assert len(corpus) == 0, "A new corpus must be of length 0"


def test_add_msg_before_date_range() -> None:
    """Test adding LogMsg before date range."""
    corpus = Corpus(date(2020, 1, 20), date(2020, 1, 31))
    msg = LogMsg(date(2020, 1, 19), "This is a test string.")
    corpus.append(msg)
    corpus.append(msg, predict=True)
    assert len(corpus) == 0, "Length of corpus must be of length 0"


def test_add_msg_after_date_range() -> None:
    """Test adding LogMsg after date range."""
    corpus = Corpus(date(2020, 1, 20), date(2020, 1, 30))
    msg = LogMsg(date(2020, 1, 31), "This is a test string.")
    corpus.append(msg)
    corpus.append(msg, predict=True)
    assert len(corpus) == 0, "Length of corpus must be of length 0"


def test_add_single_normal_msg() -> None:
    """Test adding normal LogMsg."""
    corpus = Corpus(date(2020, 1, 30), date(2020, 2, 2))
    msg = LogMsg(date(2020, 1, 30), "This is a test string.")
    corpus.append(msg)
    assert len(corpus) == 1, "Length of corpus must be of length 1"


def test_add_single_normal_msg_with_predict() -> None:
    """Test adding normal LogMsg."""
    corpus = Corpus(date(2020, 1, 30), date(2020, 2, 2))
    msg = LogMsg(date(2020, 1, 30), "This is a test string.")
    corpus.append(msg)
    corpus.append(msg, predict=True)
    assert len(corpus) == 2, "Length of corpus must be of length 2"


def test_adding_multiple_msgs() -> None:
    """Test that corpus can contain multiple LogMsg."""
    corpus = Corpus(date(2020, 1, 30), date(2020, 2, 2))
    msg1 = LogMsg(date(2020, 1, 30), "This is a test string 1.")
    msg2 = LogMsg(date(2020, 2, 1), "This is a test string 2.")
    msg3 = LogMsg(date(2020, 2, 2), "This is a test string 3.")
    corpus.append(msg1)
    corpus.append(msg2)
    corpus.append(msg3)
    assert len(corpus) == 3, "Length of corpus must be of length 3"


def test_adding_duplicate_msg() -> None:
    """Test that corpus can contain multiple duplicate LogMsg."""
    corpus = Corpus(date(2020, 1, 30), date(2020, 2, 2))
    msg1 = LogMsg(date(2020, 1, 30), "This is a test string 1.")
    msg2 = LogMsg(date(2020, 2, 1), "This is a test string 1.")
    corpus.append(msg1)
    corpus.append(msg2)
    assert len(corpus) == 2, "Length of corpus must be of length 2 (duplicate LogMsg)"


def test_distinct_lemmas() -> None:
    """Test that corpus contains the correct number of distinct lemmas."""
    corpus = Corpus(date(2020, 1, 30), date(2020, 2, 2))
    msg = LogMsg(date(2020, 1, 30), "This is a this test string test.")
    corpus.append(msg)
    assert len(corpus.distinct_words) == 2, "The number of distinct lemmas must be 2"


def test_no_lemmas_if_status_success() -> None:
    """Test corpus contains no lemmas if the status is success."""
    corpus = Corpus(date(2020, 1, 30), date(2020, 2, 2))
    msg = LogMsg(date(2020, 1, 30), "This is a test string, success.")
    corpus.append(msg)
    assert len(corpus.distinct_words) == 0, "The number of distinct lemmas must be 0"


def test_no_lemmas_if_status_failed() -> None:
    """Test corpus contains no lemmas if the status is failed."""
    corpus = Corpus(date(2020, 1, 30), date(2020, 2, 2))
    msg = LogMsg(date(2020, 1, 30), "This is a test string, failure.")
    corpus.append(msg)
    assert len(corpus.distinct_words) == 0, "The number of distinct lemmas must be 0"


def test_no_lemmas_if_status_missing() -> None:
    """Test corpus contains no lemmas if the status is 'missing'."""
    corpus = Corpus(date(2020, 1, 30), date(2020, 2, 2))
    msg = LogMsg(date(2020, 1, 30), "This is a test string (saknas)")
    corpus.append(msg)
    assert len(corpus.distinct_words) == 0, "The number of distinct lemmas must be 0"


def test_no_lemmas_if_status_error() -> None:
    """Test corpus contains no lemmas if the status is 'error'."""
    corpus = Corpus(date(2020, 1, 30), date(2020, 2, 2))
    msg = LogMsg(date(2020, 1, 30), "This is a test string (error)")
    corpus.append(msg)
    assert len(corpus.distinct_words) == 0, "The number of distinct lemmas must be 0"


def test_length_for_predict_corpus_with_single_error_logmsg() -> None:
    """Test length of predict corpus if status is 'error'."""
    corpus = Corpus(date(2020, 1, 30), date(2020, 2, 2))
    msg = LogMsg(date(2020, 1, 30), "This is a test string (error)")
    corpus.append(msg, predict=True)
    assert len(corpus) == 1, "The length of corpus must be 1"


def test_length_for_predict_corpus_with_single_success_logmsg() -> None:
    """Test length of predict corpus if status is 'success'."""
    corpus = Corpus(date(2020, 1, 30), date(2020, 2, 2))
    msg = LogMsg(date(2020, 1, 30), "This is a test string (succeeded)")
    corpus.append(msg, predict=True)
    assert len(corpus) == 1, "The length of corpus must be 1"


def test_message_id_for_multiple_msg() -> None:
    """Test that corpus stores the message id for multiple LogMsg."""
    corpus = Corpus(date(2023, 3, 10), date(2023, 3, 31))
    msg1 = LogMsg(date(2023, 3, 30), "This is a test string 1.")
    msg2 = LogMsg(date(2023, 3, 31), "This is a test string 2.")
    corpus.append(msg1)
    corpus.append(msg2)
    assert len(corpus.messages) == 2, "Instance variable messages must be of length 2"
    assert isinstance(corpus.messages[0], LogMsg), "The first message in corpus must have type LogMsg"
    assert isinstance(corpus.messages[1], LogMsg), "The second message in corpus must have type LogMsg"
