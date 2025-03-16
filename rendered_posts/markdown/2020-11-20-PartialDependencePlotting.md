Isaac Flath

AboutBlogRSS

Subscribe

AboutBlogRSS

Subscribe

# Partial Dependence Plotting

UA guide for showing the benefits and technique to Partial Dependence Plotting

## Contents

What is Partial Dependence Plotting?Why use it?ExamplesLogistic RegressionWhat now?

# What is Partial Dependence Plotting?

Traditional analysis looks at historical data and describes it. The data is fixed and you are looking to make a prediction based on the data. The person is responsible for finding **and** understanding the interactions. This gets exponentially more difficult in higher dimensional problems.

Partial Dependence Plotting looks at a prediction, and modifies the data to see how it effects the prediction. The **model** is fixed and you are looking to understand interactions by modifying the data. **The model is responsible for finding the interactions and the person just focuses on understanding the interactions the model found.**

In this way, you can think of Descriptive Analysis as a method of analysis that focuses on understanding the data, where Partial Dependence Plotting is a method of analysis that focuses on understanding the predictions. They can accomplish similar goals but approach it differently.

# Why use it?

One of the **biggest challenges of statistics is that is requires a person to make a series of assumptions**. Whether you are doing p-values, or correlations, or any other test you are typically making some assumption. Maybe you have to assume the data follows a particular distribution, maybe you have to assume that your variables are independent, or maybe you assume your variables and linearly dependent. Whatever the case may be, you typically have to make some assumption - and if you make the wrong assumptions you can get incorrect findings. It's for this reason that P-Values are not recommended for use in validating results by the majority of the top statisticians (including the American Statistical Association), despite the fact that most business analysts use them heavily in their analysis and decision making.

This leads me to the first advantage of the partial dependence plotting approach, which is a big differentiator between statistics and data science. **With partial dependence plotting you are testing through experimentation rather than through descriptive statistics and statistical testing**. For example a Neural Network can approximate any function, whether that is linear, exponential, logistic, or any other shape with any number of interactions. So I can use that, measure the accuracy, and understand the uncertainty of my analysis with no assumptions about distribution, co-linearity, or type of function.

A second advantage is that if you have sufficient data for your analysis, but it's not all the right type due to some data imbalance you can leverage all the data for the analysis. For example, if you are looking at customer churn you likely have many more accounts that did not churn than accounts that did churn. With descriptive statistics you will look at the few churned accounts and see what they have in common. With a model driven approach, **you can look at accounts that did not churn and see what changes you could make to those accounts that makes them more likely to churn.**

A third advantage is that the output you are seeing is the prediction. **The prediction of the future is often what is of value** , so if the goal is to make some decision based on what is likely to happen in the future **keeping that in the forefront is ideal.**

# Examples

Enough jibber-jabber. Let's look at some examples. We'll start simple and get a little bit more complicated over time.

### Logistic Regression

Let's create a model on the classic iris dataset and fit a logistic regression to it. When we do this, we see we have a mean accuracy score of about 97%.

[code]

    from sklearn.datasets import load_iris
    from sklearn.linear_model import LogisticRegression
    X, y = load_iris(return_X_y=True)
    clf = LogisticRegression(max_iter=500).fit(X, y)
    clf.score(X, y)
[/code]

0.9733333333333334

That's pretty good, but we don't really know what the important variables are. Let's experiment a bit.

for our first 2 rows of data we can see the model predicts 0, which is the correct answer. What changes could we make to the data to make it predict something else?

[code]

    X[0:2, :]
[/code]

array([[5.1, 3.5, 1.4, 0.2], [4.9, 3. , 1.4, 0.2]])

[code]

    clf.predict(X[0:2, :])
[/code]

array([0, 0])

Let's test adding to each of the columns and see if any change the predictions. Our original prediction is [0,0], so anything different tells us something

[code]

    for i in range(0,X_temp.shape[1]):
      X_temp = X[0:2, :].copy()
      X_temp[:,i] = X_temp[:,i]+20
      print('New Values')
      print(X_temp)
      print('Prediction')
      print(clf.predict(X_temp))
[/code]

[code]

    New Values
    [[25.1  3.5  1.4  0.2]
     [24.9  3.   1.4  0.2]]
    Prediction
    [1 1]
    New Values
    [[ 5.1 23.5  1.4  0.2]
     [ 4.9 23.   1.4  0.2]]
    Prediction
    [0 0]
    New Values
    [[ 5.1  3.5 21.4  0.2]
     [ 4.9  3.  21.4  0.2]]
    Prediction
    [2 2]
    New Values
    [[ 5.1  3.5  1.4 20.2]
     [ 4.9  3.   1.4 20.2]]
    Prediction
    [2 2]
    
[/code]

That's interesting. We can see that adding to the 1st, 3rd, and 4th variable made our model make a different prediction. Since we know our model was about 97% accurate, we know this is meaningful. It is picking up on some trend in the data using each of these columns. Lets hone in on column 1 to see if we can understand this more.

[code]

    for i in range(-10,10):
      X_temp = X[0:2, :].copy()
      X_temp[:,0] = X_temp[:,0]+i
      if (clf.predict(X_temp) == np.array([0,0])).all():
        continue
      print('Prediction for adding ' +str(i))
      print(clf.predict(X_temp))
[/code]

[code]

    Prediction for adding 4
    [0 1]
    Prediction for adding 5
    [1 1]
    Prediction for adding 6
    [1 1]
    Prediction for adding 7
    [1 1]
    Prediction for adding 8
    [1 1]
    Prediction for adding 9
    [1 1]
    
[/code]

Great, here we see that subtracting from that value does not change the prediction, but once we start adding 4 - 5 to it changes. Our original value for these rows were 5.1 and 4.9, so it seems that when we get to the 9 - 10 range for that value it becomes more likely that it is a different class all other things equal.

This is great insight, and throughout this process we made no assumptions of co-linearity, distribution, or anything else. we just manipulated the data to see the change in predictions of the mode.

##### Neural Network

Let's try a slightly more difficult problem and use a Neural Net. I could have used an XGBoost or a random forest, or any other model choice.

The goal of this dataset is to predict whether the person makes more or less than 50K salary. This could be useful for marketing reasons if you are trying to decide what products to market to whom. Let's first look at the data.

[code]

    from fastai.tabular.all import *
    path = untar_data('ADULT_SAMPLE')
    df = pd.read_csv(path/'adult.csv')
    df.head()
[/code]

| age | workclass | fnlwgt | education | education-num | marital-status | occupation | relationship | race | sex | capital-gain | capital-loss | hours-per-week | native-country | salary  
---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---  
0 | 49 | Private | 101320 | Assoc-acdm | 12.0 | Married-civ-spouse | NaN | Wife | White | Female | 0 | 1902 | 40 | United-States | >=50k  
1 | 44 | Private | 236746 | Masters | 14.0 | Divorced | Exec-managerial | Not-in-family | White | Male | 10520 | 0 | 45 | United-States | >=50k  
2 | 38 | Private | 96185 | HS-grad | NaN | Divorced | NaN | Unmarried | Black | Female | 0 | 0 | 32 | United-States | <50k  
3 | 38 | Self-emp-inc | 112847 | Prof-school | 15.0 | Married-civ-spouse | Prof-specialty | Husband | Asian-Pac-Islander | Male | 0 | 0 | 40 | United-States | >=50k  
4 | 42 | Self-emp-not-inc | 82297 | 7th-8th | NaN | Married-civ-spouse | Other-service | Wife | Black | Female | 0 | 0 | 50 | United-States | <50k  
  
We can see we've got a lot of difference variables or both categorical and continuous to look at. Lets format this for a Neural Network and fit the model.

[code]

    to = TabularPandas(df, procs=[Categorify, FillMissing,Normalize],
                       cat_names = ['workclass', 'education', 'marital-status', 'occupation', 'relationship', 'race'],
                       cont_names = ['age', 'fnlwgt', 'education-num'],
                       y_names='salary',
                       splits=RandomSplitter(valid_pct=0.2)(range_of(df)))
    dls = to.dataloaders(bs=64)
[/code]

[code]

    learn = tabular_learner(dls, metrics=accuracy, wd=0.01)
    print(to.valid.xs.shape)
    learn.fit_one_cycle(3)
[/code]

epoch | train_loss | valid_loss | accuracy | time  
---|---|---|---|---  
0 | 0.363723 | 0.374584 | 0.823710 | 00:04  
1 | 0.350536 | 0.357585 | 0.832002 | 00:04  
2 | 0.342484 | 0.354669 | 0.831081 | 00:04
[code]

    to.y
[/code]  
  
21805 0 14537 0 1399 0 8107 0 16255 0 .. 3603 1 15429 0 3551 0 1880 0 30442 0 Name: salary, Length: 32561, dtype: int8

[code]

    df.iloc[0]
[/code]

age 49 workclass Private fnlwgt 101320 education Assoc-acdm education-num 12 marital-status Married-civ-spouse occupation NaN relationship Wife race White sex Female capital-gain 0 capital-loss 1902 hours-per-week 40 native-country United-States salary >=50k Name: 0, dtype: object

Perfect - we have pretty good accuracy and our validation set has over 6000 data points. Let's look at some features to see if we can understand what impacts our model's prediction of the individuals salary.

0 = <50K, so it correctly predicts the first row is someone the makes more than 50k. Let's see at what point the prediction switches if I reduce hours worked.

[code]

    row, clas, probs = learn.predict(df.iloc[0])
    clas
[/code]

<IPython.core.display.HTML object>

tensor(1)

[code]

    for i in range(-40,0):
      X_temp = df.iloc[0].copy()
      X_temp['hours-per-week'] = X_temp['hours-per-week']+i
      row, clas, probs = learn.predict(X_temp)
      if clas == tensor(1):
        continue
      print('Prediction for adding ' +str(i))
      print(clas)
      
    from IPython.display import clear_output
    clear_output()
[/code]

Interestingly, the model isn't convinced even if I change hours works to 0, maybe it thinks the money is passive income or comes from the husband. Let's see if we can figure that out.

[code]

    for i in df.relationship.unique():
      X_temp = df.iloc[0].copy()
      X_temp['relationship'] = i
      row, clas, probs = learn.predict(X_temp)
      print('Prediction for adding ' +str(i))
      print(clas)
[/code]

<IPython.core.display.HTML object>

[code]

    Prediction for adding  Wife
    tensor(1)
    
[/code]

<IPython.core.display.HTML object>

[code]

    Prediction for adding  Not-in-family
    tensor(0)
    
[/code]

<IPython.core.display.HTML object>

[code]

    Prediction for adding  Unmarried
    tensor(0)
    
[/code]

<IPython.core.display.HTML object>

[code]

    Prediction for adding  Husband
    tensor(0)
    
[/code]

<IPython.core.display.HTML object>

[code]

    Prediction for adding  Own-child
    tensor(0)
    
[/code]

<IPython.core.display.HTML object>

[code]

    Prediction for adding  Other-relative
    tensor(0)
    
[/code]

If we change the relationship to anything else, the model starts thinking she makes less than 50K instead of more. We can continue to experiment with individual rows, or when ready try some larger experiments across the larger dataset.

[code]

    df[['salary','age']][:1000].groupby('salary').count()
[/code]

| age  
---|---  
salary |   
<50k | 754  
>=50k | 246
[code]

    results = list()
    for i in df.relationship.unique():
      g50k, l50k = (0,0)
      df_temp = df[:1000].copy()
      df_temp['relationship'] = i
      for rownum in range(0,len(df_temp)):
        if learn.predict(df_temp.iloc[rownum])[1] == tensor(1):
          g50k += 1
        else:
          l50k += 1
        clear_output()
      results.append((i,g50k,l50k))
    results
[/code]  
  
[(' Wife', 232, 768), (' Not-in-family', 165, 835), (' Unmarried', 150, 850), (' Husband', 210, 790), (' Own-child', 161, 839), (' Other-relative', 107, 893)]

We see that changing the relationship did impact the predictions across 1000 samples. Married couples, both "Wife" and "Husband" seem to be much better off with making more than 50K per year. I wonder if their income is combined and thus being double counted. "Unmarried" and "Other-Relative" seem to be the relationship types that are least likely to make more than 50K. Keep iterating through experiment like we have been to dig deeper!

You may be asking, how do I know 1000 samples is enough? The answer - I don't. Try 1500 samples next and if the results show roughly the same thing you know you have a representative sample. This is again, the data science approach vs the statistical approach.

# What now?

Now, you keep digging like we have been. We can be completely sure that the model has found meaningful interactions. We saw that in the accuracy score. So rather than me trying to find interactions, I am trying to uncover the interactions that the model has found. Again, I am making no assumptions in regard to distribution, co-linearity, type of relationships between variable, sample size, or anything else so it allows for much safe analysis of complex datasets.

### Stay Updated

Get notified about new posts on AI, web development, and tech insights.

Subscribe

## Contents

What is Partial Dependence Plotting?Why use it?ExamplesLogistic RegressionWhat now?

## Let's Connect

GitHub

LinkedIn

Twitter

© 2025 Isaac Flath • All rights reserved

