Isaac Flath

AboutBlogRSS

Subscribe

AboutBlogRSS

Subscribe

# Gradient Descent for Linear Regression - Part 1B

Unders the hood of gradient descent for model optimization

## Contents

Why part 1B?Goal RecapSee this actually work

# Why part 1B?

I have been getting questions about the initial Gradient Descent Post. The questions boil down to "So with 2 points I can define a line, but I could already do that. What I need is to fit a line where points aren't perfect and I have lots of them. How do I use gradient descent in a more complicated problem?

This post will quickly recap the initial Gradient Descent for Linear Regression post, show that methodology applied in a google sheet so you can see each calculation, and then scale that methodology to more points.

# Goal Recap

The goal of linear regression is to find parameter values that fit a linear function to data points. The goodness of fit is measured by a cost function. The lower the cost, the better the model. Gradient Descent is a method to minimize a cost function. Gradient descent is a widely used tool and is used frequently in tuning predictive models. It’s important to understand the method so you can apply it to various models and are not limited to using black box models.

I will use the sum of squares cost function to take a predicted line and slowly change the regression coefficients until the line is a line of best fit. Here's what it looks like before and after 24 iterations of gradient descent. As you can see, after 24 iterations our predicted points are getting pretty close to a best fit. You will be able to use the method defined here to scale this to as many points as you have.

In the first blog we showed this table for how we calculate out prediction. Because we are talking about a linear problem, y = mx + b is the equation, or in calculus terms $y = \theta_0+\theta_1x$. We could take this table and expand it down to include $x_3$ all the way through $x_n$ to represent our dataset.

The tree below from the first blog illustrates how to solve for cost as well as how to improve the values of $\theta$ to minimize cost in the 2 point problem defined there. So the question is, how would we modify this tree for more points? Well, with more data points there would be more edges originating at $J$, and with more features there would be more thetas originating from the predicted values, but the same concept can be applied to these more complicated examples. Specifically, here is what I would change for a more complicated example with more features:

  1. First, we have a branch for $x^1$ and a branch for $x^2$. These branches are almost identical, other than it being for the 2nd point vs the 1st point. So the first step is to add a branch off of $J = A^1 + a^2$ for $x^3$ all the way to $x^n$.
  2. The second step is to take our formula $1/2 * (y_{pred} - y_{obs})^2$ and change $1/2$ to 1 over <# of data points>. This isn't strictly neccesary, but it makes the values of J we see a bit more intiutive and scaled.
  3. The third thing is to multiply our path values by 1 over <# of data points>. Again, this isn't strictly neccesary but it makes setting a learning rate much more intuitive rather than having to do something more complicated to scale our learning rate based on the number of points we have. As a refresher, the path value for theta 1 was $\theta_1; path;value = x^1 (y^1_{pred} - y^1_{obs}) (1) + x^2 (y^2_{pred} - y^2_{obs}) (1)$, which by multiplying values from the edges in the chart together. The path value for theta 1 will now be $\theta_1; path;value = (x^1 (y^1_{pred} - y^1_{obs}) (1) + x^2 (y^2_{pred} - y^2_{obs}) (1)) * \frac{1}{<# features>)}$. We will do that for the path value formula for $\theta_0$ as well.

Just like with 2 points, we will multiply the path value by $\alpha$, and subtract that from that $\theta$ to improve our predictions

# See this actually work

I have created a google sheet that walks through these cauculations. I strongly reccomend walking through each cell calculation and matching it up to the chart above. Star with the 2Points_Linear_Scalable tab. You can then go to the More_Points_Linear tab and see that it's the exact same formulas and calculations.

Click here for the Google Sheet

For bonus points, you can start to see what a more advanced gradient descent algorithm is on the Momentum tab. If you look through all the formulas, you will see it's almost the same thing - but instead of using just the new path value we are doing a weighted average of the path value with the previous path value.

### Stay Updated

Get notified about new posts on AI, web development, and tech insights.

Subscribe

## Contents

Why part 1B?Goal RecapSee this actually work

## Let's Connect

GitHub

LinkedIn

Twitter

© 2025 Isaac Flath • All rights reserved

