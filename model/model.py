"""Outlier model to priority order log data."""

import heapq
import logging
import math
import operator
from collections import Counter
from dataclasses import dataclass
from typing import List, Tuple

import numpy as np
from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from base import Corpus

logger = logging.getLogger(__name__)


@dataclass(order=True, frozen=True)
class PredictionResult:
    """Data container to store prediction results.

    The data stored by PredictionResult is the final data reported for each log
    message in the predict data.
    """

    priority: int
    runtime: str
    date_created: str
    msg_raw: str
    explanation: str = ""


class OutlierModel:
    """OutlierModel class.

    Class to estimate a K-means outlier model, detect outliers, detect
    failed/successful executions and to assign priority to the log message data
    aimed for prediction (predictdata).
    """

    def __init__(self, traindata: Corpus, predictdata: Corpus, runtime: str) -> None:
        """Init method for OutlierModel.

        Args:
        ----
            traindata: train data used to build the K-means model
            predictdata: predict data to classify in terms of outlier detection
            and priority assignment

        """
        self._train_vectorizer = TfidfVectorizer(
            min_df=1,
            max_df=0.95,
            use_idf=True,
            stop_words=None,
            vocabulary=traindata.distinct_words.union(predictdata.distinct_words),
        )
        self._train_kmeans_model: KMeans = KMeans()
        self._train_data = traindata
        self._train_data_text = self._train_vectorizer.fit_transform(traindata)
        self._predict_data = predictdata
        self._predict_data_text = self._train_vectorizer.transform(predictdata)
        self._runtime = runtime

    def __repr__(self) -> str:
        """Object representation.

        Returns
        -------
            A repr of the object.

        """
        return (
            f"<OutlierModel({repr(self._train_data).strip('<>')}, "
            f"{repr(self._predict_data).strip('<>')}, {self._runtime})>"
        )

    def __str__(self) -> str:
        """Object as string.

        Returns
        -------
            A print of the object.

        """
        return (
            f"An outlier model with vectorizer {self._train_vectorizer}, "
            f"train corpus {repr(self._train_data).strip('<>')} and "
            f"predict corpus {repr(self._predict_data).strip('<>')} "
            f"with model {self._train_kmeans_model}. "
            f"Runtime {self._runtime}. "
        )

    def estimate(self, min_node_size: int, min_no_clusters: int) -> None:
        """Perform K-means clustering.

        Repeated clustering as long as the smallest cluster exceeds threshold
        min_node_size. The total number of clusters returned must not be lower
        than min_no_clusters.

        Args:
        ----
            min_node_size: the size of the smallest cluster
            min_no_clusters: the lowest number of allowed clusters

        """
        iters = range(1, 101)

        k = 1  # avoid unbound error
        for k in iters:
            model = KMeans(n_clusters=k, n_init=50, random_state=392)
            model_fit = model.fit(self._train_data_text)
            inertia = model_fit.inertia_
            min_size_cluster = min(
                Counter(
                    model_fit.predict(
                        self._train_data_text,
                    ),
                ).values(),
            )
            logger.info("Fitted %s clusters", k)
            logger.info(
                "Smallest cluster size is %s with total WSSE=%s",
                min_size_cluster,
                inertia,
            )
            if min_size_cluster <= min_node_size:
                logger.info(
                    "Minimum cluster size is %s, consider to estimate %s clusters",
                    min_size_cluster,
                    k - 1,
                )
                break
        logger.info("Estimate K-Means model with %s clusters", k - 1)
        decision_k = max(min_no_clusters, k - 1)
        self._train_kmeans_model = KMeans(
            n_clusters=decision_k,
            n_init=50,
            random_state=392,
        )
        self._train_kmeans_model.fit(self._train_data_text)

    def predict(self) -> List:
        """Predict method applied to the prediction data.

        Identify failed/success status, target outliers and assign priority to
        each log message in the prediction data.

        Returns
        -------
            a collection of prediction result with set priority

        """
        # dictionary of cluster assignment and training data indices
        training_cluster_idx = {
            cluster: list(
                np.where(
                    self._train_kmeans_model.labels_ == cluster,
                )[0],
            )
            for cluster in range(self._train_kmeans_model.n_clusters)
        }

        prediction_cluster_assignments = self._train_kmeans_model.predict(
            self._predict_data_text,
        )

        logger.info(
            "There are %s data points (log messages) in the prediction corpus",
            len(self._predict_data),
        )
        logger.info(
            "There are %s cluster assignment for the predictions",
            len(prediction_cluster_assignments),
        )

        # evaluate each prediction
        evaluations = []
        for idx, pred_cluster in enumerate(prediction_cluster_assignments):
            msg_raw = self._predict_data.messages[idx].msg_raw
            logger.info("Predict data point (log message) with index %s", idx + 1)
            logger.info("Predict raw message: %s", msg_raw)

            # priority queue of prediction result
            prio_queue: List[Tuple[int, PredictionResult]] = []

            # report failure
            # - exclusion: train corpus does not include 'failure'
            if self._predict_data.messages[idx].failure():
                priority_failure = 700
                evaluation = PredictionResult(
                    priority=priority_failure,
                    runtime=self._runtime,
                    date_created=self._predict_data.messages[idx].date_created.strftime(
                        "%Y-%m-%d",
                    ),
                    msg_raw=msg_raw,
                    explanation=(
                        f"Execution status 'Failure' "
                        f"(failed jobs are always reported with priority "
                        f"{priority_failure}, independent of frequency)"
                    ),
                )
                logger.info(
                    "Execution status 'Failure'. No comparison with training data.",
                )
                evaluations.append(evaluation)
                continue

            # report success
            # - exclusion: train corpus does not include 'success'
            if self._predict_data.messages[idx].success():
                priority_success = 100
                evaluation = PredictionResult(
                    priority=priority_success,
                    runtime=self._runtime,
                    date_created=self._predict_data.messages[idx].date_created.strftime(
                        "%Y-%m-%d",
                    ),
                    msg_raw=msg_raw,
                    explanation=(
                        f"Execution status 'Success' "
                        f"(successful jobs are always reported with priority "
                        f"{priority_success})"
                    ),
                )
                logger.info(
                    "Execution status 'Success'. No comparison with training data.",
                )
                evaluations.append(evaluation)
                continue

            # compare prediction with each training data point within cluster
            training_data = training_cluster_idx[pred_cluster]
            size_train = len(training_data)
            logger.info(
                "There are %s training data points assigned to the same cluster as "
                "the current prediction",
                size_train,
            )
            num_unequal = num_equal = 0

            for training_data_idx in training_data:
                # compute cosine similarity
                cos_sim = cosine_similarity(
                    self._predict_data_text[idx],
                    self._train_data_text[training_data_idx],
                )
                if math.isclose(cos_sim, 1.0):
                    num_equal += 1
                else:
                    num_unequal += 1

            # outlier tests
            outlier_tests = {}

            outlier_tests[900] = (
                math.isclose(num_unequal / size_train, 1.0),
                (
                    f"Genuine outlier! "
                    f"Log message type unobserved during "
                    f"{self._train_data.fromdate} - "
                    f"{self._train_data.enddate}"
                ),
            )
            outlier_tests[850] = (
                0 < num_equal <= 5,
                (
                    f"Borderline outlier. "
                    f"Log message type observed {num_equal} times "
                    f"during {self._train_data.fromdate} - "
                    f"{self._train_data.enddate}"
                ),
            )
            outlier_tests[840] = (
                0 < num_equal <= 10,
                (
                    f"Rare log message. "
                    f"Log message type observed {num_equal} times "
                    f"during {self._train_data.fromdate} - "
                    f"{self._train_data.enddate}"
                ),
            )
            outlier_tests[830] = (
                0 < num_equal <= 15,
                (
                    f"Rare log message. "
                    f"Log message type observed {num_equal} times "
                    f"during {self._train_data.fromdate} - "
                    f"{self._train_data.enddate}"
                ),
            )
            outlier_tests[250] = (
                0 < num_equal <= 20,
                (
                    f"Relatively rare log message. "
                    f"Log message type observed {num_equal} times "
                    f"during {self._train_data.fromdate} - "
                    f"{self._train_data.enddate}"
                ),
            )
            outlier_tests[230] = (
                20 < num_equal <= 40,
                (
                    f"Common log message. "
                    f"Log message type observed {num_equal} times "
                    f"during {self._train_data.fromdate} - "
                    f"{self._train_data.enddate}"
                ),
            )
            outlier_tests[200] = (
                num_equal > 40,
                (
                    f"Very common log message. "
                    f"Log message type observed more than 40 times "
                    f"during {self._train_data.fromdate} - "
                    f"{self._train_data.enddate}"
                ),
            )

            for priority, value in outlier_tests.items():
                bool_expr, explanation = value
                if bool_expr:
                    evaluation = PredictionResult(
                        priority=priority,
                        runtime=self._runtime,
                        date_created=self._predict_data.messages[
                            idx
                        ].date_created.strftime("%Y-%m-%d"),
                        msg_raw=msg_raw,
                        explanation=explanation,
                    )
                    heapq.heappush(
                        prio_queue,
                        (-priority, evaluation),
                    )

            # pop the prediction result with the highest priority
            # - 'success' and 'failure' are already included in "evaluations"
            _, evaluation = heapq.heappop(prio_queue)
            evaluations.append(evaluation)
        return sorted(
            evaluations,
            key=operator.attrgetter("priority", "date_created", "msg_raw"),
            reverse=True,
        )
