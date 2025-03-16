Isaac Flath

AboutBlogRSS

Subscribe

AboutBlogRSS

Subscribe

# Tabular Modeling

How to approach tabular modeling

## Contents

Project StructureBaseline ModelsPreliminary EDAHierarchical ClusteringStandard Correlation MatricesNull Value AnalysisUnderstanding HierarchiesPredictive EDAFeature ImportanceTime ConsistencyColumn sort order Model ConstructionFeature Engineering: Model Architecture: Random Forest, XGB, NNCategorical encodingClass imbalanceEnsemble Approaches: Try a few ensemble approaches.Model TuningModel EDA

**Background:** The most common type of data in most organizations is tabular data stored in a SQL database. Tabular is also a great thing to model with because it's often already structured in a database and in a good format to start working with. The uses for tabular data range from recommendation engines, to marketing action prediction, to churn prediction, predicting fraud, and much more.

**Purpose:** The purpose of this post is to walk through the things I think about for every tabular modeling problem. I may not use every technique for every problem, and there are others that I may use on occasion - but these core ones should always be at least considered carefully (in my opinion)

**Assumptions:** In this guide I am assuming you have experience building tabular models in the past and are able to look up these techniques on your own. This will offer brief descriptions of what things are and why they are important, but won't go into the level of depth needed in this post to implement these ideas.

**What's missing from this post:** Modeling is only a small piece of the puzzle. I am not covering how to define a problem, how to do ETL processes, how to deploy and productionalize things, how to communicate to the organization so it can be used, how to build processes around it so people can leverage your model, etc.. This is only covering the modeling portion.

# Project Structure

There are 5 main phases to a tabular modeling project. Each serves it's own role.

  * **Baseline Models**
  * **Preliminary EDA**
  * **Predictive EDA**
  * **Model Construction**
  * **Mode Evaluation**

The one thing that is common in **EVERY** phase is you should be constantly iterating. Every time you learn something add it to your model and try it. Experiment with the actual model. A big mistake that people often make is building lots and lots of stuff and then trying to put it all together at the end. You should be doing small micro adjustments to your model frequently and running experiments constantly.

# Baseline Models

> #### Goals
>
>   * Validate you've created and loaded the dataset properly.
>   * Verify that what you are predicting can be predicted and understand the difficulty of the problem.
>   * Verify there's no obvious data leakage in train vs valid
>

In this phase we are making baseline models. The more complex the problem the more important this is. Get an idea of the distribution of the labels. What would your accuracy be if you predicted the dominant class all the time? How good is just predicting the average value? You can build these as formal models or if it's simple enough looking at the distribution.

From there we add a bit of complexity as we start to iterate. You can move to a linear regression, logistic regression, random forest, or something else that is easy to train and set up. Keep an eye on your metrics and look for things that are "too good to be true". If you're getting a really high score in these early phases you should verify that you don't have a data leakage problem. You do want to make sure you are doing better than just predicting the most common label. If a simple model isn't better than that, a super advance complex model probably wont help either.

During this step you should also be verifying that you have created and loaded the dataset properly (including train/valid/test splits). Do not just assume previous steps were done correctly.

You need to create your train and validation sets in this phase, which is crucial. It's a good idea to use adversarial validation to check for data leakage between your train and valid sets as well.

# Preliminary EDA

> #### Goal
>
> Identify a smaller set of columns to start working with.**

The goal of this phase is to narrow the search. Other than the obvious reasons of focusing time on higher value items, it also is important for determining feature importance later.

In the Preliminary EDA phase there's 4 main things I look at.

  1. Hierarchical clustering
  2. Standard correlation matrices
  3. Null Value Analysis
  4. Understanding hierarchies

## Hierarchical Clustering

**Are any of these columns in the same sort order (regardless of scale or actual values)?**

[code]

        
    
[/code]

Hierarchical clustering is underused and very valuable. Tree based models (RF, XGboost, Catboost, etc) do not care about scale, just sort order. A decision tree treats [1,2,3] the same as [1,20,300]. Because of that doing hierarchical clustering can help identify variables that are very similar for the purposes of modeling and is a powerful tool. Instead of doing the clustering you could just do a spearman rank correlation matrix, which is the foundation for the hierarchical clustering.

## Standard Correlation Matrices

**Are any of these columns really highly correlated with other features and can be thrown out?**

Standard Correlation Matrices can be helpful as a secondary tool to identify similar columns

## Null Value Analysis

**Are there patterns of null values where several columns are all always null as the same times?**

Analyzing where Null Values appear can be helpful in understanding how data was gathered or how processes effect the data. There also may be groups of columns that are always null at the same time and you can avoid having null columns for each column and just have 1 for the set.

## Understanding Hierarchies

Giving careful thought to what a unique record is and understanding hierarchies is important? Often there may be product hierarchies, or account hierarchies, or billing hierarchies to be aware of.

# Predictive EDA

> #### Goal
>
> Understand what features are most important and how they relate to the dependent variable.

## Feature Importance

**When building a model of remaining features, which features are most important? Are there any that are not important?**

This is where we do our feature importance. It's important to get rid of as many redundant features as possible before doing feature importance because having redundant features skews feature importance. If there are 10 identical columns, then they each show as 1/10 the predictive power as any of the individual column. You could easily filter then all out thinking none are important, when in reality you should have filtered 9 of them out. The hierarchical Clustering in the previous section removes these redundancies.

Once you have the most important features you need to do a few things. First, learn as much as you can about those columns. How is data collected? How is data updated? Is there any data leakage? What's the distribution? What's the significance to the business? The more you can learn about these important features the better.

If features are very unimportant you can cut them. Simply choose an importance threshold, cut the features and measure performance of the model. If the model still performs well then those features were not crucial.

[code]

    
[/code]

## Time Consistency

**How consistent is the predictive power of a feature over time?**

[code]

    
[/code]

Measuring time consistency is an absolute necessity. We are training on data to predict future data - and in production we will be predicting data even further in the future. If features are unstable over time we need to be aware of that, and there's a good chance we can't use those features reliably.

To measure this you can build a 1 feature model to predict the dependent variable and see if predictive power remains constant over time for that feature. You can then go to 2 feature models and do the same thing. If features are extremely predictive for a few months and then not other months, your model will need some way to determine when to rely on that and when not to. Sometimes it's in the months close to a contract start/end date but often it's just a feature we have to throw away.

## Column sort order

**How does the order in which we evaluate columns matter? If we have 2 highly correlated columns, does it matter which we choose to keep?**

[code]

    
[/code]

Earlier we threw away columns that were correlated to only keep 1. In this step you can do tests to identify which should be thrown away. If 2 columns are correlated and you threw 1 of them out, was that the right choice or should you have thrown the other out?

This often doesn't make a huge difference but it does sometimes so it is worth considering.

# Model Construction

> #### Goal
>
> Take our model and everything we have learned and put it together into a final model

This is constructing the final model using everything we knew previously. There are a lot of components. I will touch on a few of the most critical ones.

## Feature Engineering:

**How can we engineer and encode features to improve feature importance and/or time consistency?**

I wish there was great universal guidance here, but it's very domain specific. Try an idea, measure validation performance keep if it helps. Graph and visualize as much as you can and keep trying new ideas.

## Model Architecture: Random Forest, XGB, NN

Considering architecture is important. For most supervised tabular modeling you will want a Random Forest, XGBoost, or a Neural Network. Random Forest is the simplest. It is fairly accurate, fast to train and hard to screw up. It's also the model with the more interpretability so if explainability is important this can be a good option.

XGboost often gives a performance boost, but it is more sensitive to hyperparameters and overfitting so it will take a bit longer to get it tuned properly.

A Neural Network is not usually the right answer for tabular data. It might be worth trying if you have super high cardinality features, or really complex data you want to use alongside it (ie combining with text NLP model).

## Categorical encoding

The main ways to encode categorical variables are label, order, target encoding, count, and catboost. I won't cover what each of them are, but try encoding things in different ways and see if it helps!

## Class imbalance

Most problems in the real world I have run into have a class imbalance. For example predicting customer churn there are way more customers staying than churning (hopefully). Churn may account for 2% of the base. There's several things that can be done to address this, the most common approaches are using a weighted loss or over/under sampling.

In Class imbalance problems it is also important that you have a good metric. I am not discussing metrics here because the metric should have been defined before the dataset was even created.

## Ensemble Approaches: Try a few ensemble approaches.

Ensembling is common, but not always necessary. You can do cross-folding, but be aware it adds complexity at inference time. Sometimes ensembling different architectures together can be valuable.

It really depends on the business use case and how much the increase in accuracy amounts to in real-world impact.

## Model Tuning

Model tuning can give improvements as well - most libraries have a built in tuning framework. You will need to set hyperparameter ranges to try and run tests with different features. There are many guides to fine tune a model - I will do a future post on how I approach tuning. Sometimes a grid search is sufficient, but for XGBoosts I often use Bayesian Optimization with HYPEROPT.

# Model EDA

The goal of this phase is to build a model with what we found out in the first 2 and look to understand the model more deeply. Hopefully this understanding can lead to us iterating and improving out model. We may revisit any or all of previous steps as we explore.

This phase we do a lot of Partial Dependence Plotting. I won't cover the details here because they are covered in the linked post.

The second thing I use in this phase is the `treeinterpreter` and `waterfallcharts` libraries. This allows me to look at individual or groups of predictions and see the contributions of different features. For more information see Chapter 9 of the fastai book

### Stay Updated

Get notified about new posts on AI, web development, and tech insights.

Subscribe

## Contents

Project StructureBaseline ModelsPreliminary EDAHierarchical ClusteringStandard Correlation MatricesNull Value AnalysisUnderstanding HierarchiesPredictive EDAFeature ImportanceTime ConsistencyColumn sort order Model ConstructionFeature Engineering: Model Architecture: Random Forest, XGB, NNCategorical encodingClass imbalanceEnsemble Approaches: Try a few ensemble approaches.Model TuningModel EDA

## Let's Connect

GitHub

LinkedIn

Twitter

© 2025 Isaac Flath • All rights reserved

