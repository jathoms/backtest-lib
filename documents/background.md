# BACKTESTING

## 1. STRATEGY

Investing is all about _strategy_<sup>\[citation needed\]</sup>. In broad terms, backtesting is simply the process of evaluating how a strategy **would have** performed if it was used throughout some historical time-period.

Before we can do any evaluation of a strategy, we have to pin down exactly what exactly a strategy is.

Investopedia nonwithstanding, we'll define a strategy as _a behaviour informed by **data** that seeks to maximise some variable_. Otherwise it'd be random, and therefore not a strategy.

More specifically, we'll define a _trading strategy_ as a behaviour which, through trading select securities, maximises the value of a given portfolio after some time period i.e 1 year. Again, based on **data**.

This is still an extremely broad definition, as "what I had for lunch today" is data.

In order to create a useful (read: encode-able) definition of a trading strategy, we'll have to narrow down our criteria.

## 2. DATA

In a general backtesting library, we want the user's strategy to be potentially informed by **any data**. This might be something sensible like the Moving Average Convergence/Divergence (MACD) of a security, or something silly, like what the Chairman of the Fed had for breakfast that day.

In essence, we need a variable scheme that, through clever-enough application by the user, can describe any reasonable set of data in a format that can be used by the library in order to [generate trades](design.md).

For instance, if the price of AAPL __always__ increases on days where the Chairman eats an apple that day, we may pass this binary variable (ate an apple? 1/0) into our strategy and find that its weight for the AAPL security is very high. In this scenario, the problem statement becomes finding out if the Chairman ate an apple as early in the day as possible (out of scope for this library).

## 3. PREDICTION

The lingering problem is our inability to see into the future and predict next period's prices for the universe.

Fortunately, apparently Fischer Black and Robert Litterman figured it out (???).

> While modern portfolio theory is an important theoretical advance, its application has universally encountered a problem: although the covariances of a few assets can be adequately estimated, it is difficult to come up with reasonable estimates of expected returns.

> Black–Litterman overcame this problem by not requiring the user to input estimates of expected return; instead it assumes that the initial expected returns are whatever is required so that the equilibrium asset allocation is equal to what we observe in the markets. The user is only required to state how their assumptions about expected returns differ from the markets and to state their degree of confidence in the alternative assumptions. From this, the Black–Litterman method computes the desired (mean-variance efficient) asset allocation. 

As soon as I understand what's going on in here, we're in business.