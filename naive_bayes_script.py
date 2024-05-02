import math

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split

iris = load_iris()

X_train, X_test, y_train, y_test = train_test_split(iris.data[:,0:4], iris.target, test_size=0.2, train_size=0.8)
print(X_test)
print(y_test)

all_target_names = iris.target_names
all_feature_names = iris.feature_names

length_of_total_samples = len(y_train)  # get the total length of the training set

arranged_dict = {key:[] for key in range(len(all_target_names))}  # separate the training set based on their target (y)  # create keys first based on the index of the targets
for index in range(len(X_train)):
    arranged_dict[y_train[index]].append(X_train[index])
print(arranged_dict)

# get the probability of each target (y)
probability_of_each_target = {}
for key, value in arranged_dict.items():
    total_length_of_value = len(value)
    probability_of_each_target[key] = total_length_of_value/length_of_total_samples
copy_dict = probability_of_each_target
probability_of_each_target = dict(sorted(copy_dict.items()))  # sort the dict by its key
print(probability_of_each_target)

# sort the values of the feature set for each data point by the index of the feature (group sepal width all together etc.) 
grouped_features = []
print(arranged_dict)
for key, value in arranged_dict.items():
    feature_by_index_dict = {}
    for each_val in value:
        for index in range(len(each_val)):
            if index in feature_by_index_dict.keys():
                feature_by_index_dict[index].append(each_val[index])
            else:
                feature_by_index_dict[index] = [each_val[index]]
    grouped_features.append(feature_by_index_dict)

# print(grouped_features)

from statistics import mean, stdev

mean_and_standard_deviation_for_each_feature_based_on_target = {}
# find the mean and standard deviation for all features (amongst each feature of the target) to be used in the Gaussian distribution function later, and it is processed with a string of its target name to be printed nicely 
for i in range(len(grouped_features)):
    curr_target = all_target_names[i]
    each_target = []
    for each_grouped_feature in grouped_features[i]:
        curr_feature = all_feature_names[each_grouped_feature]
        target_and_feature = curr_target + " " + curr_feature
        curr_data = grouped_features[i][each_grouped_feature]
        each_target.append([mean(curr_data), stdev(curr_data)])
    mean_and_standard_deviation_for_each_feature_based_on_target[curr_target] = each_target
print(mean_and_standard_deviation_for_each_feature_based_on_target)

# use the Gaussian (normal) distribution formula to find the probability density of a given feature value since this is continuous data

def find_gaussian_distribution(datapoint, mean, standard_deviation):
    first_part = 1 / math.sqrt(2 * math.pi * standard_deviation**2)
    second_part = math.exp(-(datapoint - mean)**2 / (2 * standard_deviation**2))
    return first_part * second_part

all_predictions = []  # predictions of target name of X_test dataset

for each_dataset in X_test:
    all_gaussian_distributions = []
    for key, value in mean_and_standard_deviation_for_each_feature_based_on_target.items():
        each_gaussian_distribution = []
        for index in range(len(each_dataset)):
            each_gaussian_distribution.append(find_gaussian_distribution(each_dataset[index], value[index][0], value[index][1]))  # find the gaussian distribution for each feature based on the target
        all_gaussian_distributions.append(each_gaussian_distribution)
    # print(all_gaussian_distributions)
        
    # start to multiply all probabilities together, normalize and predict
    all_probabilities = []
    for each_gaussian_idx in range(len(all_gaussian_distributions)):
        print(all_gaussian_distributions[each_gaussian_idx])
        initial_probability = math.prod(all_gaussian_distributions[each_gaussian_idx])  # use naive bayes to find the total of all probabilities 
        total_probability = initial_probability*probability_of_each_target[each_gaussian_idx]  # must also multiply all probabilities (that means all features) with the prior probability of each class that was found from the training dataset
        all_probabilities.append(total_probability)
    print(all_probabilities)
    
    # starting to normalize probabilities
    sum_of_all_probabilities = sum(all_probabilities)
    normalized_all_probabilities = [a/sum_of_all_probabilities for a in all_probabilities]
    # print(normalized_all_probabilities)
    
    # check which probability is highest and match that datapoint with its target
    max_probability = max(normalized_all_probabilities) 
    max_probability_index = normalized_all_probabilities.index(max_probability)
    all_predictions.append(all_target_names[max_probability_index])

print(all_predictions)

# calculating the score
correct_predictions = sum(1 for true, pred in zip(y_test, all_predictions) if all_target_names[true] == pred)
accuracy = correct_predictions / len(y_test)
print(f"Accuracy: {accuracy:.2f}")
