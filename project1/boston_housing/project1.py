# Import libraries necessary for this project
import numpy as np
import pandas as pd
import visuals as vs # Supplementary code
#from sklearn.cross_validation import ShuffleSplit # was getting deprecation warning
from sklearn.model_selection import ShuffleSplit


# Pretty display for notebooks
#%matplotlib inline

# Load the Boston housing dataset
data = pd.read_csv('housing.csv')
prices = data['MEDV']
features = data.drop('MEDV', axis = 1)
    
# Success
print "Boston housing dataset has {} data points with {} variables each.".format(*data.shape)

# TODO: Minimum price of the data
minimum_price = np.min(prices)

# TODO: Maximum price of the data
maximum_price = np.max(prices)

# TODO: Mean price of the data
mean_price = np.mean(prices)

# TODO: Median price of the data
median_price = np.median(prices)

# TODO: Standard deviation of prices of the data
std_price = np.std(prices)

# Show the calculated statistics
#print "Statistics for Boston housing dataset:\n"
#print "Minimum price: ${:,.2f}".format(minimum_price)
#print "Maximum price: ${:,.2f}".format(maximum_price)
#print "Mean price: ${:,.2f}".format(mean_price)
#print "Median price ${:,.2f}".format(median_price)
print "Standard deviation of prices: ${:,.2f}".format(std_price)

# TODO: Import 'r2_score'
from sklearn.metrics import r2_score

def performance_metric(y_true, y_predict):
    """ Calculates and returns the performance score between 
        true and predicted values based on the metric chosen. """
    
    # TODO: Calculate the performance score between 'y_true' and 'y_predict'
    score = r2_score(y_true, y_predict)
    
    # Return the score
    return score

    # TODO: Import 'train_test_split'
#from sklearn.cross_validation import train_test_split
from sklearn.model_selection import train_test_split
# TODO: Shuffle and split the data into training and testing subsets
#X_train, X_test, y_train, y_test = (None, None, None, None)
X_train, X_test, y_train, y_test = train_test_split(features, prices, test_size=0.20, random_state=36)

# Success
print "Training and testing split was successful."

# Produce learning curves for varying training set sizes and maximum depths
#vs.ModelLearning(features, prices)

#vs.ModelComplexity(X_train, y_train)


# TODO: Import 'make_scorer', 'DecisionTreeRegressor', and 'GridSearchCV'

from sklearn.metrics import  make_scorer
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import GridSearchCV

def fit_model(X, y):
    """ Performs grid search over the 'max_depth' parameter for a 
        decision tree regressor trained on the input data [X, y]. """
    
    # Create cross-validation sets from the training data
    #cv_sets = ShuffleSplit(X.shape[0], n_iter = 10, test_size = 0.20, random_state = 0)
    #cv_sets = ShuffleSplit(X.shape[0], n_splits = 10, test_size = 0.20, random_state = 0)
    cv_sets = ShuffleSplit( n_splits = 10, test_size = 0.20, random_state = 0)
    print 'n_splits=10'
    # TODO: Create a decision tree regressor object
    #regressor = None
    regressor =   DecisionTreeRegressor(random_state=0)

    # TODO: Create a dictionary for the parameter 'max_depth' with a range from 1 to 10
    #params = {}
    params = {'max_depth':[1,2,3,4,5,6,7,8,9,10]}

    # TODO: Transform 'performance_metric' into a scoring function using 'make_scorer' 
    #scoring_fnc = None
    scoring_fnc = make_scorer(performance_metric)

    # TODO: Create the grid search object
    #grid = None
    grid = GridSearchCV(regressor,param_grid=params,scoring=scoring_fnc, cv=cv_sets)

    # Fit the grid search object to the data to compute the optimal model
    grid = grid.fit(X, y)

    # Return the optimal model after fitting the data
    return grid.best_estimator_

# Fit the training data to the model using grid search
reg = fit_model(X_train, y_train)

# Produce the value for 'max_depth'
print "Parameter 'max_depth' is {} for the optimal model.".format(reg.get_params()['max_depth'])