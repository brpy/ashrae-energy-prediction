# Abstract - Case Study 1

<br>
<br>

## ASHRAE - Great Energy predictor III

<br>

## Overview:

### Introduction:

Modern buildings use a lot of Electrical energy for various applications like heating, cooling, steam and electrical usage. Modern Electrical retrofits can improve the Electrical power efficiency and reduce Electrical consumption and cost. But these retrofits in the scale of a large building can be quite expensive. So, instead of paying the price for retrofits upfront; customers are billed as if the retrofits were not fit for a certain period; till the costs are recovered. Let's assume these variables to be `true_cost` and `assumed_cost` ; where `true_cost` is the actual electrical consumption after retrofits and `assumed_cost` is the cost that the building would consume if there were no retrofits. The problem at hand is we don't have the `assumed_cost` information. This could be due to not having enough historical data for the particular building.

> *"Retrofitting"* - refers to the process of updating older equipment with new technology.

But we have historical data of electrical consumption for over 1000 buildings over a year's period. The task is to predict the `assumed_cost` for a new building using the historical data provided for 1000 buildings.

This competition (<https://www.kaggle.com/c/ashrae-energy-prediction/overview>) was conducted on Kaggle by ASHRAE (American Society of Heating and Air-Conditioning Engineers) on October 15 2019.

### Business problem:

A accurate model would provide better incentives for customers to switch to retrofits. Predicting accurately will smoothen the customer onboarding process and will result in increase in customers. This is because a good model's predictions will result in customers paying the bill amount almost equal to what they would've payed if there were no retrofits; resulting in customers not having to change their expenditures. There are no strict business requirements other than model being highly accurate. Latency requirements are also not high.


Check full abstract [here](./docs/abstract.md)
