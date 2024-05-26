# Direct Marketing Analysis

## Introduction


In this project, we will ingest the data, which has no missing values and thus requires no data cleaning. We will conduct an analysis to identify the variables closely linked to customers purchasing additional services and examine how these variables affect the chances of a successful marketing strategy. Based on our findings, we will provide recommendations to better utilise these variables to increase efficiency and customer purchases.

## Data 

The original data used in this project is from a public Kaggle dataset called "Banking Dataset - Marketing Targets" and can be found [here](https://www.kaggle.com/datasets/prakharrathi25/banking-dataset-marketing-targets) 


Dataset:
* `age`: Age of the customer
* `job`: Job type
* `marital`: Marital status
* `education`: Education level
* `default`: Has credit in default? (yes/no)
* `balance`: Average yearly balance in euros
* `housing`: Has housing loan? (yes/no)
* `loan`: Has personal loan? (yes/no)
* `contact`: Contact communication type (cellular/telephone)
* `day`: Last contact day of the month
* `month`: Last contact month of year
* `duration`: Last contact duration in seconds
* `campaign`: Number of contacts performed during this campaign
* `pdays`: Number of days since the client was last contacted from a previous campaign (numeric, -1 means client was not previously contacted)
* `previous`: Number of contacts performed before this campaign
* `poutcome`: Outcome of the previous marketing campaign (unknown/success/failure/other)
* `y`: Has the client subscribed a term deposit? (yes/no)`

## Code

The code is designed to be flexible and can be applied to new data received in the future.

* `direct_marketing_analysis_with_commentary.ipynb`: A Jupyter notebook containing the complete report, including code, outputs, commentary, insights, and recommendations.
* `direct_marketing_analysis_no_commentary.ipynb`: A Jupyter notebook featuring the full report with only the code and outputs.
* `direct_marketing_presentation.pptx`: A concise presentation summarising insights and actions to improve efficiency and increase customer purchases.
* `direct_marketing_analysis.py`: A Python executable script that can be run directly in the terminal. It outputs statistical information to the terminal and displays visual plots in separate windows, with each plot appearing sequentially after the previous window is closed.
