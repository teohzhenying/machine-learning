import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

df = pd.read_csv('dataR2.csv')
all_possible_features = df.columns.tolist()[0:-1]  # Index(['Age', 'BMI', 'Glucose', 'Insulin', 'HOMA', 'Leptin', 'Adiponectin', 'Resistin', 'MCP.1', 'Classification'], dtype='object')
X = df.iloc[:,:-1]
y = df['Classification']

best_features = []

X_train, X_test, y_train, y_test = train_test_split(X, y)  # split data into training and testing sets

def find_best_feature():  # function to find best feature for every loop 
    
    best_score = 0
    best_feature = None
    
    for x in all_possible_features:  # loop through the entire subset of features (X) available
        
        if x not in best_features:  # check if the current feature has already been chosen as one of the features in the final subset
            
            current_features = best_features + [x]  # train the model using the already chosen feature and the current feature (if there has not yet been any chosen features, train the model with only the current feature)
            curr_X = X_train[current_features]
            lr = LogisticRegression(max_iter=900, random_state=42)
            lr.fit(curr_X, y_train)
            
            train_score = lr.score(curr_X, y_train)  # get the score of the model using the current subset of features. This checks how well the fitted model in the training set can predict its label.
            
            curr_X_test = X_test[current_features]  # use the same LogisticRegression model on the testing set
            
            test_score = lr.score(curr_X_test, y_test)  # get the score of the testing set with the current feature subset. This checks for the model's generalization ability by seeing how well the fitted model can predict the label for the data points om the testing set.
            
            score = 0.7*train_score + 0.3*test_score  # using this ratio or similar to get the total score
            
            if score > best_score:  # if the model has a score better than the current best score
                best_score = score
                best_feature = x  # the best feature will be the current feature 
                
    return best_feature

for idx in range(0, 3):  # for this dataset, the model will be using a subset of 3 features
    best_features.append(find_best_feature())  # the best_features list will be appended with the best feature chosen at each loop until there are 3 features

print(best_features)
