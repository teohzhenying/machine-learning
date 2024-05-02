release_dates = [1897, 1998, 2000, 1948, 1962, 1950, 1975, 1960, 2017, 1937, 1968, 1996, 1944, 1891, 1995, 1948, 2011, 1965, 1891, 1978]

def min_max_normalize(lst):
    """
    normalize the data given in lst so every value is between 0 and 1
    :param lst: a list of data
    :return: a list of data, normalized
    """
    minimum = min(lst)
    maximum = max(lst)
    normalized = []
    
    for x in lst:
        normalize_x = (x - minimum)/(maximum - minimum)
        normalized.append(normalize_x)
        
    return normalized


print(min_max_normalize(release_dates))

from movies import movie_dataset, movie_labels


def distance(movie1, movie2):
    squared_difference = 0
    for i in range(len(movie1)):
        squared_difference += (movie1[i] - movie2[i]) ** 2
    final_distance = squared_difference ** 0.5
    return final_distance

def classify(unknown, dataset, labels, k):
    """
    finding the nearest neighbors
    :param unknown: the data point you want to classify
    :param dataset: the dataset you are using to classify it
    :param k: the number of neighbors you are interested in
    :return: 
    """
    distances = []
    for title in dataset:
        data = dataset[title]
        distance_to_point = distance(data, unknown)
        # Adding the distance and point associated with that distance
        distances.append([distance_to_point, title])
    print("distances: ", distances)
    print('\n')
    neighbors = sorted(distances)[0:k]  # get k nearest neighbors
    print("neighbors: ", neighbors)
    print('\n')

    num_good = 0
    num_bad = 0
    for movie in neighbors:
        title = movie[1]
        label = movie_labels[title]
        if label == 0:
            num_bad += 1
        if label == 1:
            num_good += 1

    print("num_bad, num_good: ", num_bad, num_good)
    print('\n')

    # classify unknown movie by number of num_good and num_bad neighbors
    if num_good > num_bad:
        return 1
    return 0

chosen_movie = input("Choose a movie: ")

all_movies = [x.lower() for x in movie_dataset.keys()]

# datapoint of my_movie: the movie's budget, the movie's runtime, the year the movie was released
movie_budget = input("Enter chosen_movie budget:")
movie_runtime = input("Enter chosen_movie runtime:")
movie_year_released = input("Enter chosen_movie year released:")
my_movie = [int(movie_budget), int(movie_runtime), int(movie_year_released)]  # input data for chosen_movie here 

# normalize the movie data
normalized_my_movie = min_max_normalize(my_movie)

# knn model
print(classify(normalized_my_movie, movie_dataset, movie_labels, 5))

while chosen_movie.lower() in all_movies:
    chosen_movie = input("Choose a movie: ")
print("Movie chosen: ", chosen_movie)


print(classify([0.4, 0.2, 0.9], movie_dataset, movie_labels, 5))


