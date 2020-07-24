# Poisson-model
An rather old and naive approach for predictions of soccer results. Poisson is a probabily distribution used for counting

Here we consider goals scored by each team as independent events, which of course it is not. To make a better model we have to consider a 
teams ability to take control of the game early and score the first goal. Then we have to evaluate the opponent team ability to bounce back/
the ability of the team leading to hold on to the lead. ect.

Also added a part to the pymc3 distribution family file to check how easy it was to do, since some people say a Weibull distribution is
more appropriate to use in this case. (For me it was in anaconda\\(env\myenv\\)Lib\site-packages\pymc3\glm\families.py)

```
# CHANGE THIS 
__all__ = ['Normal', 'StudentT', 'Binomial', 'Poisson', 'NegativeBinomial']
# TO
__all__ = ['Normal', 'StudentT', 'Binomial', 'Poisson', 'NegativeBinomial', 'DiscreteWeibull']


# ADD THIS
class DiscreteWeibull(Family):
    link = logit
    likelihood = pm_dists.DiscreteWeibull
    parent = 'q'
    priors = {'q': pm_dists.HalfCauchy.dist(beta=1/10, testval=1.),
              'beta': pm_dists.HalfCauchy.dist(beta=1/10, testval=1.)}

```
