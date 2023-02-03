Panel Forecasts of Country-Level Covid-19 Infections
Laura Liu, Hyungsik Roger Moon, and Frank Schorfheide
This version: April 2020

The code in this replication package illustrates how to produce country-level density forecasts and generate figures in Appendix A (forecast origin: 2020-04-18).

1. Main program:
- main.m: the main program, it performs sample selection and estimates the model
- param_heter.m: implement the parametric random effects predictor with heteroskedasticity

2. "data_results" folder:
- daily_data.txt: daily country/region-level data on confirmed cases, recovered cases, deaths, and active infections (from 2020-01-22 to 2020-04-18). The data set is constructed from the COVID-19 the data repository operated by the Center for Systems Science and Engineering (CSSE) at Johns Hopkins University (https://github.com/CSSEGISandData/COVID-19).
- Intermediate results and output figures will be stored in this folder as well.

3. "tools" folder:
3.1. Calculation
- trandn.m: generate random vectors from a truncated normal distribution (Botev, 2016)

3.2. Graphs
- jbfill.m: plot shaded graphs
- graph_out.m: setup graph format and save the graph as a .png file

3.3. Forecasts
- in_out_smpl_draws.m: produce in-sample one-step-ahead predictions and out-of-sample density forecasts based on MCMC draws of parameters
- fcst_plot.m: generate density forecast figures in Appendix A

NOTE: The MATLAB code was written for and tested on MATLAB versions from R2016b to R2020a. For earlier versions of MATLAB, errors with message "Matrix dimensions must agree" may show, which can be fixed by applying explicit matrix expansion, i.e., repmat().





NOTE: The MATLAB code was written for and tested on MATLAB versions from R2016b to R2019a. For earlier versions of MATLAB, errors with message "Matrix dimensions must agree" may show, which can be fixed by applying explicit matrix expansion, i.e., repmat().