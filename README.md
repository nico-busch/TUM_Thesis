# TUM_Thesis

This repository contains the code for my master's thesis titled "Recurrent Neural Networks for Prescriptive Analytics in Commodity Procurement" including:

* Deep Neural Networks (MLP, RNN, LSTM) implemented in PyTorch for forecasting (PredNet) and prescription (PresNet) under a customized cost-sensitive loss function
* Data-driven optimization models based on regularized regression implemented in Gurobi
* Regularized regression models implemented in scikit-learn
* Univariate forecasting models implemented in R (fable)
* Numerical studies based on Monte-Carlo simulation for different price processes
* Empirical studies in cooperation with an industry partner
* Visualizations of commodity price curves, model diagnostics, feature importance, etc.

### Abstract

A structural shift in the commodity markets and growing price volatility have drastically increased the potential of financial hedging instruments to generate actual business value for commodity-purchasing firms. Motivated by a collaboration with a multi-industry technology company, this thesis studies how firms can utilize the emerging technology of deep learning (DL) to optimize their commodity procurement in the presence of price uncertainty with both spot and forward purchasing options. In a departure from earlier work on prescriptive analytics, we employ recurrent neural networks (RNNs), a class of DL algorithms that have started to dominate the research landscape in financial time series analysis. Based on best practices in time series forecasting and classification, we propose a novel RNN architecture that, instead of predicting future prices, directly prescribes executable hedging decisions under a highly adaptive opportunity-cost-sensitive loss function. Specifically, by leveraging multivariate financial and economic feature data, the model produces maximum likelihood estimates of the probability that a given forward contract offers the optimal procurement source. We conduct extensive numerical and empirical studies that demonstrate how our proposed model can exhibit both non-linearity and temporal dynamic behavior in the data without requiring the a-priori specification of the underlying price process. Compared to our industry partner’s legacy procurement policy, our approach realizes savings of €1.2 million (5.2%) over three years of backtesting.