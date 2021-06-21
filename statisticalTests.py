import pandas as pd
import plotly.express as px
import statsmodels.formula.api as sm
from statsmodels.compat import lzip
import statsmodels.stats.api as sms

SP5002 = pd.read_csv('C:\\Users\\siane\\Python\\Project\\SP500_data.csv')
RealPrice = SP5002["Real Price"]


def describe_data(SP5002):
    index = SP5002["SP500"]
    include = ['object', 'float', 'int']
    desc = index.describe(include=include)
    print(desc)
    print(SP5002.info())
    print(SP5002['Consumer Price Index'].describe())
    return SP5002


def regress_bp(SP5002):
    results = sm.ols(formula="SP500 ~ Dividend + Earnings + RealPrice",
                     data=SP5002).fit()
    print(results.summary())
    return results


def breusch_pagan_test(results):
    # Breusch-Pagan test for Heteroscedasticity
    test = sms.het_breuschpagan(results.resid, results.model.exog)
    print("")
    names = ['Breusch Pagan Statistics', 'p-value',
             'f-value', 'f p-value']
    lzip(names, test)
    bp_results = pd.DataFrame([names, test])
    print(bp_results)
    return bp_results


def whites_test(results):
    # White's Test for Heteroscedasticity
    test = sms.het_white(results.resid, results.model.exog)
    names = ['White Statistic', 'p-value',
             'f-value', 'f p-value']
    lzip(names, test)
    white_results = pd.DataFrame([names, test])
    print(white_results)


def robust_se_OLS(SP5002):
    robust = sm.ols(formula="SP500 ~ Dividend + Earnings + RealPrice",
                    data=SP5002).fit(cov_type='HC1')
    print(robust.summary())
    hypotheses = '(Dividend =0), (RealPrice =0)'
    f_test = robust.f_test(hypotheses)
    print("")
    print("F TEST")
    print(f_test)
    print("SSR")
    print(robust.ssr)
    return robust


def plot(SP5002):
    fig = px.scatter(SP5002, x="Real Dividend", y="SP500", title='Dividends on SP500')
    fig.show()
