import pandas as pd
import plotly.express as px
import statsmodels.formula.api as sm
from statsmodels.compat import lzip
import statsmodels.stats.api as sms

SP5002 = pd.read_csv('C:\\Users\\siane\\Python\\Project\\SP500_data.csv')


# Function to graph the line of best fit for dividend on SP500 index
def lobf(SP5002):
    # effect of dividend on the SP500 index
    graph2 = px.scatter(SP5002, x='Dividend', y='SP500', trendline='ols')
    graph2.show()
    return graph2


# Multiple Linear Regression and the Breusch-Pagan Statistic of that regression
def regress_bp(SP5002):
    Y = SP5002["SP500"]
    BetaHAT1 = SP5002["Dividend"]
    BetaHAT2 = SP5002["Earnings"]
    BetaHAT3 = SP5002["Consumer Price Index"]
    BetaHAT4 = SP5002["Long Interest Rate"]
    results = sm.ols(formula="Y ~ BetaHAT1 + BetaHAT2 + BetaHAT3 + BetaHAT4", data=SP5002).fit()
    print(results.summary())
    names = ['Lagrange multiplier statistic', 'p-value',
             'f-value', 'f p-value']
    test = sms.het_breuschpagan(results.resid, results.model.exog)
    print("")
    lzip(names, test)
    bp_results = pd.DataFrame([names, test])
    print(bp_results)
    return results


# Important regression output which I could use for analysis
def important_values(results):
    r2 = results.rsquared
    f = results.fvalue
    ssr = results.ssr
    important_reg_output = pd.DataFrame([[r2, f, ssr]], columns=['R-Squared', 'F-Statistic', 'SSR'])
    print("")
    print(important_reg_output)
    return important_reg_output
