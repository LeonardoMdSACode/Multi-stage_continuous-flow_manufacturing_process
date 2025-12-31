# Multi-stage Continuous-Flow Manufacturing Process

This repository contains experiments and code for modeling and predicting outputs of a **multi-stage continuous-flow manufacturing process** using machine learning. The goal is to develop predictive models that accurately estimate process outputs from sensor and time-series data in a real industrial process.

This problem is fundamentally a **multi-output regression** with temporal dependencies and cascading stages:

* Stage 1 outputs serve as inputs for Stage 2, forming a **cascade learning problem**.
* Both classical and neural models are explored to handle the structure and dependencies in the data.

---

## Repository Structure

```
Multi-stage_continuous-flow_manufacturing_process/
├── SkLearn_Multi-stage.ipynb          ← Classical ML models (scikit-learn)
├── PyTorch_Multi-stage.ipynb          ← PyTorch model implementation
├── TensorFlow_Multi-stage.ipynb       ← TensorFlow/Keras model implementation
├── continuous_factory_process.csv     ← Main dataset (continuous time series)
├── README.md                          ← This file
```

---

## Problem Overview

This project addresses a **multi-stage regression problem** where:

1. **Stage 1 outputs** are predicted from sensor and time features.
2. **Stage 2 outputs** depend on both the original features and the predictions from Stage 1.

Key challenges in this task include:

* **Cascaded prediction error propagation** — errors in Stage 1 impact Stage 2.
* **Temporal and lagged dependencies** — process history influences the current state.
* **Multi-output predictions** — both stages produce several measurement targets.

---

## Notebooks Summary

### 1. **SkLearn_Multi-stage.ipynb**

This notebook explores **classical machine learning models** using scikit-learn:

* Linear models: Ridge, ElasticNet, ElasticNetChain
* PLS Regression
* Ensemble models: RandomForest, Gradient Boosting
* Multi-output strategies: `RegressorChain`, `MultiOutputRegressor`

Key features:

* Time-based train-test split
* Feature engineering with time and lag/delta features
* Evaluation using median RMSE across multiple outputs

This notebook serves as the baseline and provides insights into conventional regression performance before deep learning approaches.

---

### 2. **PyTorch_Multi-stage.ipynb**

This notebook implements a **deep neural network** in PyTorch:

* Residual Multi-Layer Perceptron architecture for multi-output regression
* Custom scikit-learn compatible wrapper (`BaseEstimator`, `RegressorMixin`)
* Cascade training: Stage 2 uses OOF Stage 1 predictions

Important elements:

* PyTorch MLP with residual connections
* Training loops with `AdamW` optimizer
* Stage-by-stage metrics

This notebook demonstrates how deep learning can be integrated into the existing pipeline while keeping the cascade intact.

---

### 3. **TensorFlow_Multi-stage.ipynb**

This notebook parallels the PyTorch version, implemented in **TensorFlow / Keras**:

* Residual MLP architecture with additive blocks
* Custom wrapper for sklearn compatibility
* Cascade training logic identical to other notebooks

TensorFlow is used here to show an alternative deep learning framework with identical data flow.

---

## Features and Preprocessing

Across all notebooks, the following feature engineering is applied:

* **Time features**:

  * `hour`, `dayofweek`, `month`, `hour_sin`, `hour_cos`
* **Lag features**:

  * 1- and 2-step lagged Stage 1 outputs
* **Delta features**:

  * First-order differences of Stage 1 outputs

Temporal features help capture cyclical dynamics, and lag/delta features help encode dynamics in prior states.

---

## Metrics

All models are evaluated using **Median RMSE** across multiple outputs:

```python
median_rmse = np.median(
          root_mean_squared_error(y_true, y_pred, multioutput="raw_values")
)
```

This metric is robust to outliers among different target dimensions.

---

## Notes and Recommendations

* Classical models like ElasticNetChain often outperform naïve deep models on structured tabular data unless the architecture explicitly captures temporal dependencies.
* Cascade training requires careful simulation of prediction noise (e.g., out-of-fold predictions) to avoid leakage.
* Transitional approaches (hybrid or residual corrections) may yield better Stage 2 results.
* Investigate sequence models (RNN / LSTM / GRU / TCN) if Stage 2 dynamics are strongly history-dependent.

---

## Acknowledgements

This project was developed to explore regression modeling in multi-stage industrial processes where upstream predictions feed downstream models — a common scenario in manufacturing analytics.
(https://www.kaggle.com/datasets/supergus/multistage-continuousflow-manufacturing-process?resource=download&select=continuous_factory_process.csv)

---

## License

## MIT License
