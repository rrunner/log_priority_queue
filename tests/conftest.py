"""Test fixtures."""

from datetime import date
from typing import Tuple

import pandas as pd
import pytest

from base import Corpus, LogMsg
from model import OutlierModel


@pytest.fixture()
def logmsg_normal() -> LogMsg:
    """Create test data."""
    day = date(2023, 2, 13)
    test_string = "System X is down</TD><TD>2023-02-13 04:00</TD><TD>PUBLISH Time Dimensions   (12 of 12)</TD><TD>"
    return LogMsg(day, test_string)


@pytest.fixture()
def logmsg_success() -> LogMsg:
    """Create test data."""
    day = date(2023, 2, 13)
    test_string = "GEM Load Complete</TD><TD>2023-02-13 04:00</TD><TD>PUBLISH Time Dimensions   (12 of 12)</TD><TD>Succeeded"
    return LogMsg(day, test_string)


@pytest.fixture()
def logmsg_failure() -> LogMsg:
    """Create test data."""
    day = date(2023, 2, 12)
    test_string = "HRSERVICE - freeze</TD><TD>2023-02-12 13:00</TD><TD>check if there are dates in  (1 of 1)</TD><TD>Failed"
    return LogMsg(day, test_string)


@pytest.fixture()
def fail_texts() -> LogMsg:
    """Failure texts."""
    day = date(2019, 8, 18)
    test_string = (
        "fail failed failure failured FAIL FAILED FAILURE FAILURED "
        "(fail) (failed) (failure) (failured) (FAIL) (FAILED) "
        "(FAILURE) (FAILURED) saknas SAKNAS Saknas error "
        "ERROR Error (saknas) (SAKNAS) (Saknas) (error) "
        "(ERROR) (Error)"
    )
    return LogMsg(day, test_string)


@pytest.fixture()
def success_texts() -> LogMsg:
    """Success texts."""
    day = date(2019, 6, 10)
    test_string = (
        "success succeed succeeded SUCCESS SUCCEED SUCCEEDED "
        "(success) (succeed) (succeeded) (SUCCESS) (SUCCEED) "
        "(SUCCEEDED)"
    )
    return LogMsg(day, test_string)


@pytest.fixture()
def create_train_predict_corpus() -> Tuple:
    corpus_train = Corpus(date(2023, 3, 1), date(2023, 4, 1))
    corpus_predict = Corpus(date(2023, 4, 2), date(2023, 4, 2))
    # train data
    for _ in range(10):
        msg = LogMsg(date(2023, 3, 10), "System X. File import completed. All good.")
        corpus_train.append(msg)
    for _ in range(10):
        msg = LogMsg(
            date(2023, 3, 10),
            (
                "QlikSense flow data is loaded."
                "Reports will created automatically based on PUBLISH."
            ),
        )
        corpus_train.append(msg)
    # predict data
    for _ in range(1):
        msg = LogMsg(
            date(2023, 4, 2), "System X. File import issue. This can be critical."
        )
        corpus_predict.append(msg, predict=True)
    return corpus_train, corpus_predict


@pytest.fixture()
def create_excel_file(tmp_path):
    data = {
        "DateTimeCreated": [
            "2023-04-17 07:15:06.193",
            "2023-04-17 07:15:06.193",
            "2023-04-17 07:15:06.193",
            "2023-04-17 09:00:09.393",
            "2023-04-17 09:00:09.393",
            "2023-04-17 09:00:09.393",
            "2023-04-17 09:00:09.393",
            "2023-04-17 09:00:09.393",
            "2023-04-17 09:00:09.393",
            "2023-04-17 09:00:09.393",
            "2023-04-17 09:00:09.393",
            "2023-04-17 09:00:09.393",
            "2023-04-17 09:00:09.393",
            "2023-04-17 09:00:09.393",
            "2023-04-17 09:00:09.393",
            "2023-04-17 09:00:09.393",
            "2023-04-18 07:15:15.290",
            "2023-04-18 07:15:15.290",
            "2023-04-18 07:15:15.290",
            "2023-04-18 07:15:15.290",
            "2023-04-18 07:15:15.290",
            "2023-04-18 07:15:15.290",
            "2023-04-18 07:15:15.290",
            "2023-04-18 07:15:15.290",
            "2023-04-18 07:15:15.290",
            "2023-04-18 07:15:15.290",
            "2023-04-18 07:15:15.290",
            "2023-04-18 07:15:15.290",
        ],
        "NotificationMessage": [
            "SSIS Server Maintenance J</TD><TD>2023-04-17 00:00</TD><TD>SSIS Server Max Version Per Project Main   (2 of 2)</TD><TD>Succeeded",
            "VO Load complete</TD><TD>2023-04-16 10:40</TD><TD>QS Reload   (30 of 30)</TD><TD>Succeeded",
            "AD and UR extract</TD><TD>2023-04-17 05:17</TD><TD>CMD QS   (5 of 5)</TD><TD>Succeeded",
            "DatabaseIntegrityCheck - </TD><TD></TD><TD></TD><TD>",
            "EOT Load Complete</TD><TD></TD><TD></TD><TD>",
            "BGR Load Complete</TD><TD></TD><TD></TD><TD>",
            "STAT Load 01. Ålder </TD><TD></TD><TD></TD><TD>",
            "HR KOLL Load complete</TD><TD></TD><TD></TD><TD>",
            "sp_purge_jobhistory</TD><TD></TD><TD></TD><TD>",
            "GR Run QS Trigger</TD><TD></TD><TD></TD><TD>",
            "Data warehouse statis</TD><TD></TD><TD></TD><TD>",
            "BILL Yearly Clean</TD><TD></TD><TD></TD><TD>",
            "HR 01 STAGE</TD><TD></TD><TD></TD><TD>",
            "GEM Load Complete</TD><TD>2023-04-17 04:00</TD><TD>PUBLISH Time Dimensions   (12 of 12)</TD><TD>Succeeded",
            "MailFunction</TD><TD>2023-04-17 07:15</TD><TD>BIF MailNotification Send HTML MorningJo   (3 of 3)</TD><TD>Succeeded",
            "KF Load Complete</TD><TD>2023-04-17 01:03</TD><TD>Trigger QS   (34 of 34)</TD><TD>Succeeded",
            "FRF IMAS Data Load Comple</TD><TD>2023-04-18 03:30</TD><TD>EDW_MAIN Trans ImasGenderData   (8 of 8)</TD><TD>Succeeded",
            "KF BAPP - W</TD><TD>2023-04-17 08:30</TD><TD>MailNotification SendQue with attach   (5 of 5)</TD><TD>Succeeded",
            "ASTA Load Complete</TD><TD>2023-04-18 01:21</TD><TD>HR-Data   (26 of 26)</TD><TD>Succeeded",
            "VOO Load complete</TD><TD>2023-04-17 09:19</TD><TD>Start Agent job VO Load complete   (14 of 14)</TD><TD>Succeeded",
            "HRSERVICE - freeze</TD><TD>2023-04-17 13:00</TD><TD>check if there are dates in (1 of 1)</TD><TD>Failed",
            "ARO</TD><TD>2023-04-18 06:03</TD><TD>PUBLISH FactOrderRow   (12 of 12)</TD><TD>Succeeded",
            "GE Load Complete</TD><TD>2023-04-18 06:24</TD><TD>MainNotification Send   (36 of 36)</TD><TD>Succeeded",
            "FRF Get Files From IMA/R</TD><TD>2023-04-18 03:00</TD><TD>ACTOR: Download from API   (7 of 7)</TD><TD>Succeeded",
            "FGV Load Complete</TD><TD>2023-04-18 06:07</TD><TD>Load PUBLISH (17 of 17)</TD><TD>Succeeded",
            "BK Load Complete</TD><TD>2023-04-17 10:00</TD><TD>BK CheckData - If failed on step 3, file   (3 of 17)</TD><TD>Failed",
            "BKG Load Complete</TD><TD>2023-04-17 14:00</TD><TD>BKG CheckData - If failed on step 3, fi   (3 of 18)</TD><TD>Failed",
            "SSIS Server Maintenance J</TD><TD>2023-04-18 00:00</TD><TD>SSIS Server Max Version Per Project Main   (2 of 2)</TD><TD>Succeeded",
        ],
    }
    df = pd.DataFrame(data)
    file_path = tmp_path / "test.xlsx"
    df.to_excel(file_path, index=False)
    return file_path


@pytest.fixture()
def priority():
    corpus_train = Corpus(date(2023, 3, 1), date(2023, 4, 1))
    corpus_predict = Corpus(date(2023, 4, 2), date(2023, 4, 2))
    # train data
    for _ in range(100):
        msg = LogMsg(date(2023, 3, 10), "System X. File import completed. All good.")
        corpus_train.append(msg)
    for _ in range(5):
        msg = LogMsg(date(2023, 3, 31), "System 5 times. Testing!")
        corpus_train.append(msg)
    for _ in range(10):
        msg = LogMsg(date(2023, 3, 31), "System 10 times. Testing, testing $$$")
        corpus_train.append(msg)
    for _ in range(14):
        msg = LogMsg(date(2023, 3, 31), "hard psuedo text, 14")
        corpus_train.append(msg)
    # predict data
    for _ in range(1):
        msg = LogMsg(date(2023, 4, 2), "Some kind of outlier. This can be an issue.")
        corpus_predict.append(msg, predict=True)
    for _ in range(1):
        msg = LogMsg(date(2023, 4, 2), "System 5 times. Testing!")
        corpus_predict.append(msg, predict=True)
    for _ in range(1):
        msg = LogMsg(date(2023, 4, 2), "System 10 times. Testing, testing $$$")
        corpus_predict.append(msg, predict=True)
    for _ in range(2):
        msg = LogMsg(date(2023, 4, 2), "hard psuedo text, 14")
        corpus_predict.append(msg, predict=True)
    om = OutlierModel(corpus_train, corpus_predict, "2023-04-02 21:00:00")
    om.estimate(min_node_size=130, min_no_clusters=1)
    predictions = om.predict()
    return predictions


@pytest.fixture()
def success_failure():
    corpus_train = Corpus(date(2023, 3, 1), date(2023, 4, 1))
    corpus_predict = Corpus(date(2023, 4, 2), date(2023, 4, 2))
    # train data
    for _ in range(100):
        msg = LogMsg(date(2023, 3, 10), "System X. File import completed. All good.")
        corpus_train.append(msg)
    # predict data
    for _ in range(1):
        msg = LogMsg(date(2023, 4, 2), "System XYZ, this is a success")
        corpus_predict.append(msg, predict=True)
    for _ in range(1):
        msg = LogMsg(date(2023, 4, 2), "System ZYX, this is an error")
        corpus_predict.append(msg, predict=True)
    om = OutlierModel(corpus_train, corpus_predict, "2023-04-02 21:00:00")
    om.estimate(min_node_size=130, min_no_clusters=1)
    predictions = om.predict()
    return predictions
