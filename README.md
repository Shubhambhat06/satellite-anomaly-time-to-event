# Satellite Anomaly Time-to-Event Prediction

<p align="center">

  <img src="https://img.shields.io/badge/Python-3.x-blue?style=for-the-badge&logo=python" alt="Python">
  <img src="https://img.shields.io/badge/Scikit--Learn-ML-orange?style=for-the-badge&logo=scikit-learn" alt="Scikit-Learn">
  <img src="https://img.shields.io/badge/Pandas-Data%20Processing-150458?style=for-the-badge&logo=pandas" alt="Pandas">
  <img src="https://img.shields.io/badge/NumPy-Numerical%20Computing-013243?style=for-the-badge&logo=numpy" alt="NumPy">
  <img src="https://img.shields.io/badge/Domain-Space%20Weather-purple?style=for-the-badge" alt="Space Weather">

</p>

<h3 align="center">
Satellite Anomaly Prediction as a Time-to-Event Regression Problem
</h3>

<p align="center">
A machine learning pipeline for estimating the time until the next recorded satellite anomaly using orbital characteristics, solar activity, and temporal information.
</p>

---

# Table of Contents

- [Overview](#overview)
- [Motivation](#motivation)
- [Problem Statement](#problem-statement)
- [Research Context](#research-context)
- [Project Objectives](#project-objectives)
- [System Overview](#system-overview)
- [Dataset Sources](#dataset-sources)
- [Data Engineering Pipeline](#data-engineering-pipeline)
- [Time-to-Event Formulation](#time-to-event-formulation)
- [Feature Engineering](#feature-engineering)
- [Final Dataset](#final-dataset)
- [Machine Learning Models](#machine-learning-models)
- [Evaluation Methodology](#evaluation-methodology)
- [Experimental Results](#experimental-results)
- [Interpretation of Results](#interpretation-of-results)
- [Why Relative Error Is Large](#why-relative-error-is-large)
- [Current Evaluation Limitation](#current-evaluation-limitation)
- [Repository Structure](#repository-structure)
- [Installation](#installation)
- [Dataset Setup](#dataset-setup)
- [Running the Pipeline](#running-the-pipeline)
- [Data Validation](#data-validation)
- [Running Individual Models](#running-individual-models)
- [Reproducing the Experiments](#reproducing-the-experiments)
- [Git Workflow](#git-workflow)
- [Technical Details](#technical-details)
- [Design Decisions](#design-decisions)
- [Limitations](#limitations)
- [Future Work](#future-work)
- [Research Extensions](#research-extensions)
- [Reference Work](#reference-work)
- [Citation](#citation)
- [Acknowledgements](#acknowledgements)
- [Author](#author)
- [License](#license)

---

# Overview

Satellites operate in an environment influenced by orbital dynamics, solar activity, radiation, thermal conditions, atmospheric effects, and other physical factors.

These conditions can contribute to satellite anomalies, which may range from transient operational issues to more significant spacecraft events.

Traditional anomaly prediction is often formulated as a classification problem:

> Will an anomaly occur?

This project approaches the problem differently.

Instead of predicting only whether an anomaly will occur, the project formulates satellite anomaly prediction as a **Time-to-Event (TTE)** problem:

> **How many days remain until the next recorded satellite anomaly?**

The project combines historical satellite anomaly records with:

- Orbital characteristics
- Solar activity
- Temporal information

and evaluates multiple machine learning approaches for predicting the resulting time-to-event value.

---

# Motivation

A binary anomaly prediction system provides limited temporal information.

For example:

```text
Anomaly predicted: YES
