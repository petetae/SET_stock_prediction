# SET_stock_prediction
### Table of Contents

1. [Overview](#overview)
2. [Installation](#installation)
3. [Project files](#project_files)
4. [Project details](#project_details)
5. [Results](#results)
6. [Local setup instructions](#local_setup)
7. [Licensing, Authors, and Acknowledgements](#licensing)

## Overview <a name="overview"></a>
Project relating to the stock market prediction and analysis in Thailand using stock data, google trend keyword interest, and covid19 statistics in Thailand.

The main problem that we are looking to explore and solve is in the stock market domain. Recent trends, in Thailand, have shown that stock market prices have been affected by the increase in covid19 statistics and this also has been influenced by the google search information. Here, we look to explore whether these have an impact in predicting the stock information or not and how we can optimise our models to yield the highest performance.

This project originated as an idea and an exploration to use various data sources to help predict the stock market data. 

The data set/source we used are as follows
StarfishX - Library to fetch the Thailand stock information
https://pypi.org/project/starfishX/ 
Google trends data - Library that allows us to fetch the search keyword interests
https://pypi.org/project/pytrends/ 
Covid19 statistics data Thailand - API call fetching JSON data
https://covid19.ddc.moph.go.th/api/Cases/timeline-cases-all 

Based on these datasets, we need to clean them for analysis.

More details can be found in the attached report or from the following medium blog 
- https://pete-taecha.medium.com/can-we-predict-thailand-stock-with-google-trends-and-covid19-stats-2b5ecf0712f4

## Installation <a name="installation"></a>

1. Download all files
2. Run the notebook proj_capstone_pete_3_predict_results.ipynb

In case the some functions do not work, we suggest you create a virtual environment and then install the following libraries:
python3 -m venv proj_xxx
source proj_xxx/bin/activate
Pip install IPython
Pip install mplfinance
Pip install starfishX
Pip install jupyter notebook
pip install pytrends
Pip install plotly
Pip install nltk -> noit used?
Pip install sqlalchemy
Pip install openpyxl
Pip install statsmodels
Pip install pandas_ta
Pip install prophet
Pip install keras
Pip install tensorflow

### Project files <a name="project_files"></a>
Main project files to refer to in this project includes
- proj_capstone_pete_1_dataexplore.ipynb
- proj_capstone_pete_2_train_model_optimise.ipynb
- proj_capstone_pete_3_predict_results.ipynb

The rest of te files are the data temporary information generated.

Report of this project is named as
- Udacity Data Science Nanodegree Capstone project - Pete v1

### Project details <a name="project_details"></a>
Project required multiple process and steps. For deep details of the project, please refer to the report file.

Overview of the project done
1. Gather the data sources (stock data, google trends data, covid19 stats data)
2. Explore, and clean the data for prepping
3. Create model to predict the stocks information
4. Optimise the model for finalisation

Learnings are mainly on the whole development flow along with the different algorithms used and optimising the model along with creating a valid training and prediction testing dataset.

## Results<a name="results"></a>
Results can be found in the following notebook.
- proj_capstone_pete_3_predict_results.ipynb

For more information, check the Jupyter notebook files or contact the project's author.

## Licensing, Authors, Acknowledgements<a name="licensing"></a>

Udacity Data Science Nanodegree program.
https://www.udacity.com/course/data-scientist-nanodegree--nd025

