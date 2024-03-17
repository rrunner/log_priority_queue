"""Base classes used to process log data.

Requires a Swedish pipeline/lemmatizer/word dictionary etc.

Download as:
>> python -m spacy download en_core_web_lg
"""

import re
from datetime import date
from typing import Iterator, List, Set, Tuple

import spacy
from spacy.tokens.doc import Doc

# word dictionary
NLP = spacy.load("en_core_web_lg")

# regex
RE_SUCCESS = re.compile(r"(?i)\bsucce[ss|ed|eded]+\b")
RE_FAILURE = re.compile(r"(?i)\bfail[ure|ed|ured]*\b")
RE_MISSING = re.compile(r"(?i)\bsaknas\b")
RE_ERROR = re.compile(r"(?i)\berror\b")
RE_DATE = re.compile(r"\d+-\d+-\d+\s+\d+:\d+")
RE_OF = re.compile(r"\(\d+\s*of\s*\d+\)")
RE_TD = re.compile(r"<[\/]?TD>")


class LogMsg:
    """Data cleanse and NLP pre-process a log message.

    Attributes
    ----------
        date_created: the date log message is created
        msg_raw: the original log message
        lemmas: the lemmas contained in the log message
        lemmas_text: the lemmas concatenated

    """

    def __init__(self, date_created: date, message: str) -> None:
        """Init method for LogMsg.

        Args:
        ----
            date_created: the date log message is created
            message: the original log message

        """
        self.date_created = date_created
        self.msg_raw = message
        self._doc: Doc = NLP(self.msg_clean(self.msg_raw))
        self.lemmas: Tuple = self.msg_to_lemmas(self._doc)
        self.lemmas_text: str = " ".join(self.lemmas)

    def __repr__(self) -> str:
        """Object representation.

        Returns
        -------
            A repr of the object.

        """
        return (
            f"<LogMsg(date_created={self.date_created!r}, "
            f"message={self.msg_raw!r})>"
        )

    def __str__(self) -> str:
        """Object as string.

        Returns
        -------
            A print of the object.

        """
        return (
            f"Date created: {self.date_created}\n"
            f"Raw: {self.msg_raw}\n"
            f"Lemmas: {self.lemmas}\n"
        )

    def __len__(self) -> int:
        """Length of object.

        Returns
        -------
            The number of lemmas contained by the object.

        """
        return len(self.lemmas)

    @staticmethod
    def msg_clean(message: str) -> str:
        """Clean raw message.

        Args:
        ----
            message: the date log message is created

        Returns:
        -------
            A cleansed log message.

        """
        message_clean = re.sub(RE_TD, " ", message)
        message_clean = re.sub(RE_OF, " ", message_clean)
        message_clean = re.sub(RE_DATE, " ", message_clean)
        message_clean = re.sub(RE_FAILURE, " ", message_clean)
        message_clean = re.sub(RE_MISSING, " ", message_clean)
        message_clean = re.sub(RE_ERROR, " ", message_clean)
        message_clean = re.sub(RE_SUCCESS, " ", message_clean)
        return message_clean.lower()

    @staticmethod
    def msg_to_lemmas(doc: Doc) -> Tuple:
        """Produce a collection of cleaned token lemmas.

        Args:
        ----
            doc: A Doc document obtained from NLP that contains useful NLP
            data.

        Returns:
        -------
            Cleansed collection of lemmas.

        """
        cleansed_lemmas = []
        for token in doc:
            if not token.is_punct and not token.is_space and not token.is_stop:
                cleansed_lemmas.append(token.lemma_)
        return tuple(cleansed_lemmas)

    def success(self) -> bool:
        """Verify success status on raw log message.

        Returns
        -------
            Boolean True if log message contains any signs of 'success'.

        """
        cond_success = [
            re.search(RE_SUCCESS, self.msg_raw),
        ]
        match_success = any(cond_success)
        if match_success:
            return True
        return False

    def failure(self) -> bool:
        """Verify failed status on raw log message.

        Returns
        -------
            Boolean True if log message contains any signs of 'failure'.

        """
        cond_failure = [
            re.search(RE_FAILURE, self.msg_raw),
            re.search(RE_MISSING, self.msg_raw),
            re.search(RE_ERROR, self.msg_raw),
        ]
        match_failure = any(cond_failure)
        if match_failure:
            return True
        return False


class Corpus:
    """Corpus class that contains processed log messages.

    Log messages are added to corpus conditional on date and status. See
    the append method.

    Attributes
    ----------
        messages: collection of (complete) log messages
        fromdate: the start date of the corpus
        enddate: the end date of the corpus

    """

    def __init__(self, fromdate: date, enddate: date) -> None:
        """Init method for Corpus.

        Args:
        ----
            fromdate: the start date of the corpus
            enddate: the end date of the corpus

        """
        self._corpus: List[str] = []
        self.messages: List[LogMsg] = []
        self.fromdate = fromdate
        self.enddate = enddate
        self.distinct_words: Set = set()

    def __len__(self) -> int:
        """Length of object.

        Returns
        -------
            the number of lemmas_text (one for each LogMsg)

        """
        return len(self._corpus)

    def __repr__(self) -> str:
        """Object representation.

        Returns
        -------
            A repr of the object.

        """
        return f"<Corpus(fromdate={self.fromdate!r}, enddate={self.enddate!r})>"

    def __str__(self) -> str:
        """Object as string.

        Returns
        -------
            A print of the object.

        """
        return (
            f"Corpus instance with {len(self)} messages, "
            f"with {len(self.distinct_words)} distinct words, "
            f"spanning the period {self.fromdate} - {self.enddate}"
        )

    def __iter__(self) -> Iterator:
        """Object iter method.

        Returns
        -------
            Return the iterator associated with the instance attribute
            self._corpus.

        """
        return self._corpus.__iter__()

    def append(self, message: LogMsg, predict: bool = False) -> None:
        """Append method.

        Add log message to an internal collection, and cleansed lemma text to
        the corpus.

        Args:
        ----
            message: a processed log message (LogMsg)
            predict: boolean True/False for data aimed for prediction/training

        """
        if not predict:
            # conditions to add log messages to train corpus
            cond = [
                not message.success(),
                not message.failure(),
                self.fromdate <= message.date_created <= self.enddate,
            ]
        else:
            # conditions to add log messages to predict corpus
            cond = [
                self.fromdate <= message.date_created <= self.enddate,
            ]
        if all(cond):
            self.messages.append(message)
            self._corpus.append(message.lemmas_text)
            for lemma in message.lemmas:
                self.distinct_words.add(lemma)
