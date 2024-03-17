## Log message priority queue application

### Introduction

This application defines a priority queue to system logs by utilizing
Unsupervised Machine Learning (UML) and Natural Language Processing (NLP)
techniques.

The application objective is to classify log messages, in terms of rules and
outlier characteristics, to enable prioritisation of log messages generated by
a system.

### Usage

In the terminal:

1. Activate the virtual environment

Activate virtual environment and install dependencies using the
requirements.txt file.

2. Run the application

The application accepts excel file and database table input.

```
    python log_priority_queue.py --help
```

### High-level methodology description

A basic TF-IDF vectorizer is applied to the training data, and the same
vectorizer is applied to transform the prediction data, in order to map the
textual data into points in a coordinate system, e.g. each log message becomes
a vector with the dimensionality set by the number of unique words in the
training and prediction data.

The training data is fit by using the unsupervised K-means learning approach,
where the number of clusters are allowed to increase as long as each cluster
contains a certain number of log messages. The model is trained each time the
application is run. By fitting the model on each run, decommissioned
systems/batch load jobs etc. are excluded from training by design.

An outlier is defined as a prediction data point that deviates from all
training data points in terms of the cosine similarity.

The idea is, on the basis of rules and outlier statistics, to assign each
prediction data point a priority such that total batch of prediction data can
be arranged by priority. Outliers and 'failure' items are assigned a high
priority. Repeated log messages and 'success' items are assigned a lower
priority.

### Technical details

#### Logging

The application includes logging. Each run produces a logfile that is stored in
subfolder /log.


#### Testing

The application includes a test suite using `pytest`.
