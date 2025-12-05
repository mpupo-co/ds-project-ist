# ds-project-ist @ IST Data Science course 25/26 

## 1\. Overview
The project consists of performing the first iteration of the KDD process, when training
a set of models over two distinct datasets. Data
profiling, data preparation, modeling, and evaluation steps performed for each task, which are
[classification](/01.Classification/) and [forecasting](/02.Forecasting/).
In both situations, the goal is not only to describe the best models learned, but also to
understand the impact of the available options on the performance of the produced models.

## 2\. Classification
The datasets for the classification task in this project were collected from the Kaggle platform.
 - Security domain – Traffic Accidents
    - classification file = traffic_accidents.csv, target = CRASH_TYPE
    - [description](https://www.kaggle.com/datasets/oktayrdeki/traffic-accidents)
 
 - Economy domain – Flight Status Prediction
    - classification file = combined_flights_2022.csv, target = CANCELLED
    - [description](https://www.kaggle.com/datasets/robikscube/flight-delay-dataset-20182022)

## 3\. Forescasting
The datasets for the forecasting task were collected from the same domains as the data used for classification.
- Security domain – Traffic Prediction
    - classification file = TrafficTwoMonth.csv, target = TOTAL
    - [description](https://www.kaggle.com/datasets/hasibullahaman/traffic-prediction-dataset)
- Economy domain – Global Economic Indicators
    - classification file = economic_indicators_dataset_2010_2023.csv, target = INFLATION RATE (USA)
    - [description](https://www.kaggle.com/datasets/heidarmirhajisadati/global-economic-indicators-dataset-2010-2023)

## Code notes

In order to run the scripts, run them from the parent directory:
```
pwd: ~/ds-project-ist/.
cmd: python3 lab2/dimensionality_correlation.py
```
