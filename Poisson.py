

import pandas as pd
import numpy as np
import pymc3 as pm
import matplotlib.pyplot as plt
import seaborn
from scipy.stats import poisson,skellam
import re
import seaborn as sns
import statsmodels.api as sm
import statsmodels.formula.api as smf






norske_resultater = pd.read_csv('NOR.csv')
norske_resultater = norske_resultater.loc[norske_resultater['Season'] == 2012]
norske_resultater = norske_resultater[['Home','Away','HG','AG']]
norske_resultater = norske_resultater.rename(columns={'HG': 'HomeGoals', 'AG': 'AwayGoals'})
norske_resultater = norske_resultater[:-2]

print(norske_resultater.head())

# Formler for utregning av diverse statestikker ved bruk av pandas DataFrame
formulas = {'Hjemme_Maal_Forskjell' : 'df.Hjemme_Maal - df.Hjemme_Innsluppet', 
            'Borte_Maal_Forskjell' : 'df.Borte_Maal - df.Borte_Innsluppet', 
            'Maal' : 'df.Hjemme_Maal + df.Borte_Maal', 
            'Innsluppet' : 'df.Hjemme_Innsluppet + df.Borte_Innsluppet', 
            'Maal_Forskjell' : 'df.Maal - df.Innsluppet', 
            'Poeng' : 'df.Hjemme_Poeng + df.Borte_Poeng', 
            'Kamper_Spilt' : 'df.Hjemme_Kamper + df.Borte_Kamper', 
           }
def update(df,formulas):
           for k, v in formulas.items():
              df[k] = pd.eval(v)

            
# TABELL
def tabell(resultater):
    Lag = np.unique([resultater['Home'].unique(), resultater['Away'].unique()])
    Tabell = {'Possisjon': np.zeros(len(Lag), dtype=np.int8),
              'Lag': sorted(Lag) ,
              'Kamper_Spilt': np.zeros(len(Lag), dtype=np.int8),
              'Poeng': np.zeros(len(Lag), dtype=np.int8), 
              'Maal': np.zeros(len(Lag), dtype=np.int8),
              'Innsluppet': np.zeros(len(Lag), dtype=np.int8), 
              'Maal_Forskjell': np.zeros(len(Lag), dtype=np.int8), 
              'Hjemme_Kamper': np.zeros(len(Lag), dtype=np.int8), 
              'Hjemme_Poeng': np.zeros(len(Lag), dtype=np.int8), 
              'Hjemme_Maal': np.zeros(len(Lag), dtype=np.int8), 
              'Hjemme_Innsluppet': np.zeros(len(Lag), dtype=np.int8), 
              'Hjemme_Maal_Forskjell': np.zeros(len(Lag), dtype=np.int8),
              'Borte_Kamper': np.zeros(len(Lag), dtype=np.int8),
              'Borte_Poeng': np.zeros(len(Lag), dtype=np.int8), 
              'Borte_Maal': np.zeros(len(Lag), dtype=np.int8), 
              'Borte_Innsluppet': np.zeros(len(Lag), dtype=np.int8), 
              'Borte_Maal_Forskjell': np.zeros(len(Lag), dtype=np.int8),
             }
    Tabell = pd.DataFrame(data=Tabell)
    
    for k in range(len(resultater)):

        # Maal
        Tabell.loc[Tabell['Lag'] == resultater.iloc[k]['Home'], ['Hjemme_Maal']] += resultater.iloc[k]['HomeGoals']
        Tabell.loc[Tabell['Lag'] == resultater.iloc[k]['Away'], ['Borte_Maal']] += resultater.iloc[k]['AwayGoals']
        Tabell.loc[Tabell['Lag'] == resultater.iloc[k]['Home'], ['Hjemme_Innsluppet']] += resultater.iloc[k]['AwayGoals']
        Tabell.loc[Tabell['Lag'] == resultater.iloc[k]['Away'], ['Borte_Innsluppet']] += resultater.iloc[k]['HomeGoals']

        # Kamper
        Tabell.loc[Tabell['Lag'] == resultater.iloc[k]['Home'], ['Hjemme_Kamper']] += 1
        Tabell.loc[Tabell['Lag'] == resultater.iloc[k]['Away'], ['Borte_Kamper']] += 1

        # Poeng
        if resultater.iloc[k]['HomeGoals'] > resultater.iloc[k]['AwayGoals']:
            Tabell.loc[Tabell['Lag'] == resultater.iloc[k]['Home'], ['Hjemme_Poeng']] += 3
        if resultater.iloc[k]['HomeGoals'] == resultater.iloc[k]['AwayGoals']:
            Tabell.loc[Tabell['Lag'] == resultater.iloc[k]['Home'], ['Hjemme_Poeng']] += 1
            Tabell.loc[Tabell['Lag'] == resultater.iloc[k]['Away'], ['Borte_Poeng']] += 1           
        if resultater.iloc[k]['HomeGoals'] < resultater.iloc[k]['AwayGoals']:
            Tabell.loc[Tabell['Lag'] == resultater.iloc[k]['Away'], ['Borte_Poeng']] += 3
        
    update(Tabell, formulas)
    Tabell.sort_values(by=['Poeng'], inplace = True, ascending=False)
    Possisjon = {'Possisjon' : np.array(range(len(Lag)))+1}
    update(Tabell, Possisjon)
#     print(Tabell)

    
    
tabell(norske_resultater)




# construct Poisson  for each mean goals value
poisson_pred = np.column_stack([[poisson.pmf(i, norske_resultater.mean()[j]) for i in range(8)] for j in range(2)])
# plot histogram of actual goals
plt.hist(norske_resultater[['HomeGoals', 'AwayGoals']].values, range(9),
         alpha=0.7, label=['Home', 'Away'],density=True, color=["#FFA07A", "#20B2AA"])

# add lines for the Poisson distributions
pois1, = plt.plot([i-0.5 for i in range(1,9)], poisson_pred[:,0],
                  linestyle='-', marker='o',label="Home", color = '#CD5C5C')
pois2, = plt.plot([i-0.5 for i in range(1,9)], poisson_pred[:,1],
                  linestyle='-', marker='o',label="Away", color = '#006400')

leg=plt.legend(loc='upper right', fontsize=13, ncol=2)
leg.set_title("Poisson           Actual        ", prop = {'size':'14', 'weight':'bold'})
plt.xticks([i-0.5 for i in range(1,9)],[i for i in range(9)])
plt.xlabel("Goals per Match",size=13)
plt.ylabel("Proportion of Matches",size=13)
plt.title("Number of Goals per Match (Tippeligaen 2012 Season)",size=14,fontweight='bold')
plt.ylim([-0.004, 0.4])
plt.tight_layout()
plt.show()






# Andel Hjemme maal vs Borte maal (Note that we consider the number of goals scored by each team to be 
# independent events (i.e. P(A n B) = P(A) P(B)). The difference of two Poisson distribution is actually 
# called a Skellam distribution.)

skellam_pred = [skellam.pmf(i,  norske_resultater.mean()[0],  norske_resultater.mean()[1]) for i in range(-6,8)]
plt.hist(norske_resultater[['HomeGoals']].values - norske_resultater[['AwayGoals']].values, range(-6,8),
         alpha=0.7, label='Actual',density=True)
plt.plot([i+0.5 for i in range(-6,8)], skellam_pred,
                  linestyle='-', marker='o',label="Skellam", color = '#CD5C5C')
plt.legend(loc='upper right', fontsize=13)
plt.xticks([i+0.5 for i in range(-6,8)],[i for i in range(-6,8)])
plt.xlabel("Home Goals - Away Goals",size=13)
plt.ylabel("Proportion of Matches",size=13)
plt.title("Difference in Goals Scored (Home Team vs Away Team)",size=14,fontweight='bold')
plt.ylim([-0.004, 0.26])
plt.tight_layout()
plt.show()





fig,(ax1,ax2) = plt.subplots(2, 1, figsize=(8,4.5))

mold_home = norske_resultater[norske_resultater['Home']=='Molde'][['HomeGoals']].apply(pd.value_counts,normalize=True)
mold_home_pois = [poisson.pmf(i,np.sum(np.multiply(mold_home.values.T,mold_home.index.T),axis=1)[0]) for i in range(8)]
trom_home = norske_resultater[norske_resultater['Home']=='Tromso'][['HomeGoals']].apply(pd.value_counts,normalize=True)
trom_home_pois = [poisson.pmf(i,np.sum(np.multiply(trom_home.values.T,trom_home.index.T),axis=1)[0]) for i in range(8)]

mold_away = norske_resultater[norske_resultater['Away']=='Molde'][['AwayGoals']].apply(pd.value_counts,normalize=True)
mold_away_pois = [poisson.pmf(i,np.sum(np.multiply(mold_away.values.T,mold_away.index.T),axis=1)[0]) for i in range(8)]
trom_away = norske_resultater[norske_resultater['Away']=='Tromso'][['AwayGoals']].apply(pd.value_counts,normalize=True)
trom_away_pois = [poisson.pmf(i,np.sum(np.multiply(trom_away.values.T,trom_away.index.T),axis=1)[0]) for i in range(8)]


ax1.bar(mold_home.index-0.4,mold_home.values.T[0],width=0.4,color="#034694",label="Molde")
ax1.bar(trom_home.index,trom_home.values.T[0],width=0.4,color="#EB172B",label="Tromso")
pois1, = ax1.plot([i for i in range(8)], mold_home_pois,
                  linestyle='-', marker='o',label="Molde", color = "#0a7bff")
pois1, = ax1.plot([i for i in range(8)], trom_home_pois,
                  linestyle='-', marker='o',label="Tromso", color = "#ff7c89")
leg=ax1.legend(loc='upper right', fontsize=8, ncol=2)
leg.set_title("Poisson                 Actual                ", prop = {'size':14, 'weight':'bold'})
ax1.set_xlim([-0.5,7.5])
ax1.set_ylim([-0.01,0.65])
ax1.set_xticklabels([])
# mimicing the facet plots in ggplot2 with a bit of a hack
ax1.text(7.65, 0.585, '                Home                ', rotation=-90,
        bbox={'facecolor':'#ffbcf6', 'alpha':0.5, 'pad':5})
ax2.text(7.65, 0.585, '                Away                ', rotation=-90,
        bbox={'facecolor':'#ffbcf6', 'alpha':0.5, 'pad':5})

ax2.bar(mold_away.index-0.4,mold_away.values.T[0],width=0.4,color="#034694",label="Molde")
ax2.bar(trom_away.index,trom_away.values.T[0],width=0.4,color="#EB172B",label="Tromso")
pois1, = ax2.plot([i for i in range(8)], mold_away_pois,
                  linestyle='-', marker='o',label="Molde", color = "#0a7bff")
pois1, = ax2.plot([i for i in range(8)], trom_away_pois,
                  linestyle='-', marker='o',label="Tromso", color = "#ff7c89")
ax2.set_xlim([-0.5,7.5])
ax2.set_ylim([-0.01,0.65])
ax1.set_title("Number of Goals per Match (Tippeligaen 2012 Season)",size=14,fontweight='bold')
ax2.set_xlabel("Goals per Match",size=13)
ax2.text(-1.55, 0.9, 'Proportion of Matches', rotation=90, size=13)
plt.tight_layout()
# plt.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=None, hspace=None)
plt.show()





# Lager en generalisert Poisson regression
goal_model_data = pd.concat([norske_resultater[['Home','Away','HomeGoals']].assign(home=1).rename(
            columns={'Home':'team', 'Away':'opponent','HomeGoals':'goals'}),
           norske_resultater[['Away','Home','AwayGoals']].assign(home=0).rename(
            columns={'Away':'team', 'Home':'opponent','AwayGoals':'goals'})])

poisson_model = smf.glm(formula="goals ~ home + team + opponent", data=goal_model_data, 
                        family=sm.families.Poisson()).fit()
print(poisson_model.summary())



def simulate_match(foot_model, homeTeam, awayTeam, max_goals=10):
    home_goals_avg = foot_model.predict(pd.DataFrame(data={'team': homeTeam, 
                                                            'opponent': awayTeam,'home':1},
                                                      index=[1])).values[0]
    away_goals_avg = foot_model.predict(pd.DataFrame(data={'team': awayTeam, 
                                                            'opponent': homeTeam,'home':0},
                                                      index=[1])).values[0]
    team_pred = [[poisson.pmf(i, team_avg) for i in range(0, max_goals+1)] for team_avg in [home_goals_avg, away_goals_avg]]
    return(np.outer(np.array(team_pred[0]), np.array(team_pred[1])))


simulate_match(poisson_model, 'Molde', 'Tromso', max_goals=4)

mold_trom = simulate_match(poisson_model, "Molde", "Tromso", max_goals=10)
# Molde win
print(np.sum(np.tril(mold_trom, -1)))
# uavgjort
print(np.sum(np.diag(mold_trom)))
# Tromso win
print(np.sum(np.triu(mold_trom, 1)))

print(norske_resultater[norske_resultater['Home']=='Molde'])





def strip_derived_rvs(rvs):
    '''Remove PyMC3-generated RVs from a list'''

    ret_rvs = []
    for rv in rvs:
        if not (re.search('_log', rv.name) or re.search('_interval', rv.name)):
            ret_rvs.append(rv)
    return ret_rvs


# PYMC3 

with pm.Model() as model:
    pm.glm.GLM.from_formula(formula="goals ~ home + team + opponent", data=goal_model_data,
                            family=pm.glm.families.DiscreteWeibull())
                            # family=pm.glm.families.Poisson())
                            # family=pm.glm.families.NegativeBinomial())

    trace = pm.sample(2000, tune=2000, cores=1)

varnames = [rv.name for rv in strip_derived_rvs(model.unobserved_RVs)]
# print(plot_traces(trace, varnames=varnames))
# pm.traceplot(trace, varnames)
# print(np.exp(pm.summary(trace, varnames=varnames)[['mean','hpd_2.5','hpd_97.5']]))
print(pm.summary(trace, varnames=varnames))
pm.plot_posterior(trace)
plt.show()


