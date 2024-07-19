# Baseball Playoff Prediction Project

## Overview

This is the final project for CMSI 5350 - Machine Learning, developed by Jose and Denali. The project focuses on baseball data between the years 1962 - 2012. Some of the baseball statistics that we have access to with our baseball.csv dataset are the following: Runs Scored (RS), Runs Against (RA), On-base-percentage (OBP), Slugging (SLG), Batting Average (BA), Opponents Slugging (OSLG), and Opponents On-base-percentage (OOBP).

## Project Objective

Our objective was to see how accurately we can predict teams making the playoffs or not making the playoffs regardless of conference with the statistics above. We identified 4 different kinds of models that we believe can help us accomplish this task:

1) Logistic Regression
2) Random Forrest 
3) Support Vector Machines
4) K-nearest neighbors 

We tested the 4 models above in both the default values/settings provided for us and also used some hypertuning for each of the models to see if we can get an improvement in accurarcy. The image below shows just one example of this with one of our models, Logistic Regression. The graph on the left shows the actual reality counts of teams making the playoffs and on the right is our model (default values) predications on those same teams. Can see a side-by-side comparison and see how our model does. 

<img width="899" alt="image" src="https://github.com/jf2024/Predicting-MLB-Playoff-Teams/assets/65199388/b9c0f2ab-f4fe-49ad-8b3a-86f3f60df9be">


## Some Libraries that helped us

### 1. GridSearchCV

We utilized GridSearchCV for hyperparameter tuning of our machine learning models. GridSearchCV systematically tests different combinations of hyperparameter values that we give it to find the optimal set, improving the performance of our models.

Learn more about [GridSearchCV](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.GridSearchCV.html).

### 2. Imputation

If you take a look at our dataset, the OOBP and OSLG statistics started being recorded in the year 1999 which means before that, there was no information on those two statistics. We decided to use imputation, specifically the `SimpleImputer` from scikit-learn, to fill in those missing values in our dataset just to give our models a more complete information.

Learn more about [Imputation in scikit-learn](https://scikit-learn.org/stable/modules/impute.html).

### 3. Seaborn

Seaborn is a powerful data visualization library in Python. We utilized Seaborn to create bar plots to showcase actual and predicted playoff appearances by baseball teams to see how our models stack up to the actual counts in reality.

Learn more about [Seaborn](https://seaborn.pydata.org/).

## How to Run the Code

### Prerequisites

Before running the code, ensure that you have the following prerequisites installed on your machine:

- Python: Make sure you have Python installed. You can download it from [python.org](https://www.python.org/downloads/).
- Git: Git is required to clone the repository. You can download it from [git-scm.com](https://git-scm.com/downloads).

### Installation

Follow these steps to set up the project:

1. **Clone the Repository:**
   Open a terminal or command prompt and run the following command to clone the repository to your local machine:

   ```bash
   git clone https://github.com/LMU-CMSI-Korpusik/project-denali-jose-ryan.git
   ```

2. **Navigate to the Project Directory:**
   Change your working directory to the project folder (wherever you cloned it):

   ```bash
   cd project-denali-jose-ryan
   ```

3. **Install Dependencies:**
   Install the required Python packages by running all of the following (install each individually):

   ```bash
   pip install pandas
   pip install scikit-learn
   pip install seaborn
   pip install matplotlib
   ```

   The commands above are necessary for some of the background computation and graphs to pop up when running the file

### Run the Project

Once the installation is complete, you can run the project using the following steps:

1. **Navigate to the Code Directory:**
   Ensure you are in the project's root directory:

   ```bash
   cd path/to/project-denali-jose-ryan
   ```

2. **Execute the Main Script:**
   Run the main script `moneyball.py` to execute the project:

   ```bash
   python moneyball.py
   ```

   This command initiates the analysis, training of machine learning models, and generates predictions based on the provided baseball dataset. 

3. **View Results:**
   Once the script completes, check the terminal for output logs, including model accuracy, hyperparameter tuning results, and ablative analysis. Additionally, visualizations, such as bar plots and feature importance plots, will be displayed. You will notice about 4 graphs that will pop out on your screen (like the one from above). The graph on the left is the actual playoff count for each team and the one on the right is our model's (the default one, not the hyperparameter) predication counts for each baseball team. You will also notice different coefficient reports for each of the models for both the default values and hypertuning values.

   Additionally, give the script some time to run everything, it might take a couple of minutes and if it does seem like its stuck, just press the 'enter' key and it should just go. 

### Acknowledgements

We would like to express our gratitude to scikit-learn and Seaborn for providing essential tools and libraries that significantly contributed to the success of our project.

Also to the following resources that helped us out: 

1. Basic Understanding of [GridSearch](https://www.linkedin.com/pulse/what-gridsearchcv-randomizedsearchcv-differences-between-cheruku/)
2. Where we obtained our [dataset](https://ocw.mit.edu/courses/15-071-the-analytics-edge-spring-2017/pages/logistic-regression/assignment-3/predicting-the-baseball-world-series-champion/)
