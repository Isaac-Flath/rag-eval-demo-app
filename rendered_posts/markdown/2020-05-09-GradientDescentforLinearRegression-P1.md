Isaac Flath

AboutBlogRSS

Subscribe

AboutBlogRSS

Subscribe

# Gradient Descent for Linear Regression - Part 1

Unders the hood of gradient descent for model optimization

## Contents

The goal of linear regression is to find parameter values that fit a linear function to data points. The goodness of fit is measured by a cost function. The lower the cost, the better the model. Gradient Descent is a method to minimize a cost function. Gradient descent is a widely used tool and is used frequently in tuning predictive models. It’s important to understand the method so you can apply it to various models and are not limited to using black box models.

As I just mentioned, gradient descent is a method to reduce a cost function. To understand how to minimize a cost function, you need to understand how cost is calculated. For this post I will be using a very simple example; linear regression with one feature, two data points, and two regression coefficients. I will use the sum of squares cost function to take the predicted line and slowly change the regression coefficients until the line passes through both points.

In this example we could easily draw a line through the points without using gradient descent, but if we had more data points this would get trickier. In the table below we can see what the data looks like that we are working with.

The tree below illustrates how to solve for cost as well as how to improve the values of $\theta$ to minimize cost. In the illustration above, $J = a^1 + a^2$ is the cost function we want to minimize. As we can see, if the regression coefficients ($\theta_0+\theta_1$) do not give a good fit, then the difference between our predicted values and observed values will be large and we will have a high cost ($J$). For low values, we will have a low cost ($J$). The figure below shows us how to calculate cost from the regression coefficients ($\theta_0$ and $\theta_1$).

The second thing this chart shows you is how to improve values of theta. We used the formulas in the boxes to evaluate $J$, so now we will use the values on the edges to improve the parameter values. Each regression coefficient has a path up to the cost function. You get a path value for each $\theta$ on the tree by multiplying the edge values along that path. For example:

$\theta_1; path;value = x^1 (y^1_{pred} - y^1_{obs}) (1) + x^2 (y^2_{pred} - y^2_{obs}) (1)$

The last step is to improve the value of $\theta$. In order to improve the value of $\theta$, we need to multiply the path value by $\alpha$, and subtract that from that $\theta$. $\alpha$ is a value that determines how large the increments will be taken during optimization. If you pick an $\alpha$ value that is is too large, you risk missing the local optima. If you choose an $\alpha$ value that is too small you will be very accurate, but it will be more computationally expensive. With more data points there would be more edges originating at $J$, and with more features there would be more thetas originating from the predicted values, but the same concept can be applied to these more complicated examples.

### Stay Updated

Get notified about new posts on AI, web development, and tech insights.

Subscribe

## Contents

## Let's Connect

GitHub

LinkedIn

Twitter

© 2025 Isaac Flath • All rights reserved

