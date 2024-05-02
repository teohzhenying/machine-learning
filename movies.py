import pandas as pd
from sklearn.model_selection import train_test_split

movie_dataset = {
    'Avatar': [0.01940156245995175, 0.4812286689419795, 0.9213483146067416],
    "Pirates of the Caribbean: At World's End": [0.02455894456664483, 0.45051194539249145, 0.898876404494382],
    'Spectre': [0.02005646812429373, 0.378839590443686, 0.9887640449438202],
    'The Dark Knight Rises': [0.020465784164507467, 0.4334470989761092, 0.9550561797752809],
    'John Carter': [0.021587310114693104, 0.3242320819112628, 0.9550561797752809],
    'Spider-Man 3': [0.021120689828849445, 0.4061433447098976, 0.898876404494382],
    'Tangled': [0.021284416244934937, 0.2150170648464164, 0.9325842696629213],
    'Avengers: Age of Ultron': [0.020465784164507467, 0.35494880546075086, 0.9887640449438202],
    'Harry Potter and the Half-Blood Prince': [0.020465784164507467, 0.39590443686006827, 0.9213483146067416],
    'Batman v Superman: Dawn of Justice': [0.020465784164507467, 0.49829351535836175, 1.0],
    'Superman Returns': [0.017109392634754826, 0.45051194539249145, 0.8876404494382022],
    'Quantum of Solace': [0.0163726237623701, 0.2354948805460751, 0.9101123595505618],
    "Pirates of the Caribbean: Dead Man's Chest": [0.018419203963438782, 0.3890784982935154, 0.8876404494382022],
    'The Lone Ranger': [0.017600571883011312, 0.3856655290102389, 0.9662921348314607],
    'Man of Steel': [0.018419203963438782, 0.36177474402730375, 0.9662921348314607],
    'The Chronicles of Narnia: Prince Caspian': [0.018419203963438782, 0.3856655290102389, 0.9101123595505618],
    'The Avengers': [0.018009887923225047, 0.4641638225255973, 0.9550561797752809],
    'Pirates of the Caribbean: On Stranger Tides': [0.020465784164507467, 0.3378839590443686, 0.9438202247191011],
    'Men in Black 3': [0.018419203963438782, 0.2354948805460751, 0.9550561797752809],
    'The Hobbit: The Battle of the Five Armies': [0.020465784164507467, 0.4334470989761092, 0.9775280898876404],
    'The Amazing Spider-Man': [0.01882852000365252, 0.39590443686006827, 0.9550561797752809],
    'Robin Hood': [0.0163726237623701, 0.4061433447098976, 0.9325842696629213],
    'The Hobbit: The Desolation of Smaug': [0.018419203963438782, 0.5085324232081911, 0.9662921348314607],
    'The Golden Compass': [0.014735359601515157, 0.2593856655290102, 0.898876404494382],
    'King Kong': [0.016945666218669334, 0.5597269624573379, 0.8764044943820225],
    'Titanic': [0.0163726237623701, 0.5358361774744027, 0.7865168539325843],
    'Captain America: Civil War': [0.020465784164507467, 0.37542662116040953, 1.0],
    'Battleship': [0.017109392634754826, 0.32081911262798635, 0.9550561797752809],
    'Jurassic World': [0.012279463360232739, 0.29692832764505117, 0.9887640449438202],
    'Skyfall': [0.0163726237623701, 0.36177474402730375, 0.9550561797752809],
    'Spider-Man 2': [0.0163726237623701, 0.33447098976109213, 0.8651685393258427],
    'Iron Man 3': [0.0163726237623701, 0.5392491467576792, 0.9662921348314607],
    'Alice in Wonderland': [0.0163726237623701, 0.24232081911262798, 0.9325842696629213],
    'X-Men: The Last Stand': [0.017191255842797574, 0.22866894197952217, 0.8876404494382022],
    'Monsters University': [0.0163726237623701, 0.22866894197952217, 0.9662921348314607],
    'Transformers: Revenge of the Fallen': [0.0163726237623701, 0.3856655290102389, 0.9213483146067416],
    'Transformers: Age of Extinction': [0.017191255842797574, 0.43686006825938567, 0.9775280898876404],
    'Oz the Great and Powerful': [0.017600571883011312, 0.3174061433447099, 0.9662921348314607],
    'The Amazing Spider-Man 2': [0.0163726237623701, 0.3583617747440273, 0.9775280898876404],
    'TRON: Legacy': [0.013916727521087684, 0.3003412969283277, 0.9325842696629213],
    'Cars 2': [0.0163726237623701, 0.2354948805460751, 0.9438202247191011],
    'Green Lantern': [0.0163726237623701, 0.2935153583617747, 0.9438202247191011],
    'Toy Story 3': [0.0163726237623701, 0.22525597269624573, 0.9325842696629213],
    'Terminator Salvation': [0.0163726237623701, 0.2764505119453925, 0.9213483146067416],
    'Furious 7': [0.015553991681942629, 0.3515358361774744, 0.9887640449438202],
    'World War Z': [0.015553991681942629, 0.2935153583617747, 0.9662921348314607],
    'X-Men: Days of Future Past': [0.0163726237623701, 0.3822525597269625, 0.9775280898876404],
    'Star Trek Into Darkness': [0.015553991681942629, 0.3242320819112628, 0.9662921348314607],
    'Jack the Giant Slayer': [0.015963307722156365, 0.2627986348122867, 0.9662921348314607],
    'The Great Gatsby': [0.00859561899830911, 0.36177474402730375, 0.9662921348314607],
    'Prince of Persia: The Sands of Time': [0.0163726237623701, 0.2696245733788396, 0.9325842696629213],
    'Pacific Rim': [0.015553991681942629, 0.32081911262798635, 0.9662921348314607],
    'Transformers: Dark of the Moon': [0.015963307722156365, 0.3993174061433447, 0.9438202247191011],
    'Indiana Jones and the Kingdom of the Crystal Skull': [0.015144675641728892, 0.2901023890784983,
                                                           0.9101123595505618],
    'Brave': [0.015144675641728892, 0.19112627986348124, 0.9550561797752809],
    'Star Trek Beyond': [0.015144675641728892, 0.2901023890784983, 1.0],
    'WALL∑E': [0.014735359601515157, 0.20819112627986347, 0.9101123595505618],
    'Rush Hour 3': [0.011460831279805265, 0.18430034129692832, 0.898876404494382],
    '2012': [0.0163726237623701, 0.4129692832764505, 0.9213483146067416],
    'Bee Movie': [0.012279463360232739, 0.18430034129692832, 0.898876404494382],
    'A Christmas Carol': [0.0163726237623701, 0.20136518771331058, 0.9213483146067416],
    'Jupiter Ascending': [0.014407906769344168, 0.30716723549488056, 0.9887640449438202],
    'The Legend of Tarzan': [0.014735359601515157, 0.24914675767918087, 1.0],
    'The Chronicles of Narnia: The Lion, the Witch and the Wardrobe': [0.014735359601515157, 0.3856655290102389,
                                                                       0.8764044943820225],
    'X-Men: Apocalypse': [0.014571633185429662, 0.3651877133105802, 1.0],
    'The Dark Knight': [0.015144675641728892, 0.3924914675767918, 0.9101123595505618],
    'Up': [0.01432604356130142, 0.20136518771331058, 0.9213483146067416],
    'Monsters vs. Aliens': [0.01432604356130142, 0.1945392491467577, 0.9213483146067416],
    'Iron Man': [0.011460831279805265, 0.3037542662116041, 0.9101123595505618],
    'Hugo': [0.013916727521087684, 0.3037542662116041, 0.9438202247191011],
    'Wild Wild West': [0.013916727521087684, 0.2354948805460751, 0.8089887640449438],
    'The Mummy: Tomb of the Dragon Emperor': [0.011870147320019002, 0.25597269624573377, 0.9101123595505618],
    'Suicide Squad': [0.01432604356130142, 0.2935153583617747, 1.0],
    'Evan Almighty': [0.01432604356130142, 0.20136518771331058, 0.898876404494382],
    'Edge of Tomorrow': [0.014571633185429662, 0.2593856655290102, 0.9775280898876404],
    'Waterworld': [0.01432604356130142, 0.47440273037542663, 0.7640449438202247],
    'G.I. Joe: The Rise of Cobra': [0.01432604356130142, 0.2764505119453925, 0.9213483146067416],
    'Inside Out': [0.01432604356130142, 0.19795221843003413, 0.9887640449438202],
    'The Jungle Book': [0.01432604356130142, 0.2354948805460751, 1.0],
    'Iron Man 2': [0.0163726237623701, 0.29692832764505117, 0.9325842696629213],
    'Snow White and the Huntsman': [0.013916727521087684, 0.3242320819112628, 0.9550561797752809],
    'Maleficent': [0.014735359601515157, 0.20477815699658702, 0.9775280898876404],
    'Dawn of the Planet of the Apes': [0.013916727521087684, 0.3174061433447099, 0.9775280898876404]
}

movie_labels = {
    'Avatar': 1,
    "Pirates of the Caribbean: At World's End": 1,
    'Spectre': 1,
    'The Dark Knight Rises': 1,
    'John Carter': 0,
    'Spider-Man 3': 0,
    'Tangled': 1,
    'Avengers: Age of Ultron': 1,
    'Harry Potter and the Half-Blood Prince': 1,
    'Batman v Superman: Dawn of Justice': 0,
    'Superman Returns': 0,
    'Quantum of Solace': 1,
    "Pirates of the Caribbean: Dead Man's Chest": 1,
    'The Lone Ranger': 0,
    'Man of Steel': 1,
    'The Chronicles of Narnia: Prince Caspian': 1,
    'The Avengers': 1,
    'Pirates of the Caribbean: On Stranger Tides': 1,
    'Men in Black 3': 1,
    'The Hobbit: The Battle of the Five Armies': 1,
    'The Amazing Spider-Man': 1,
    'Robin Hood': 0,
    'The Hobbit: The Desolation of Smaug': 1,
    'The Golden Compass': 0,
    'King Kong': 0,
    'Titanic': 0,
    'Captain America: Civil War': 1,
    'Battleship': 0,
    'Jurassic World': 1,
    'Skyfall': 1,
    'Spider-Man 2': 1,
    'Iron Man 3': 1,
    'Alice in Wonderland': 1,
    'X-Men: The Last Stand': 1,
    'Monsters University': 1,
    'Transformers: Revenge of the Fallen': 1,
    'Transformers: Age of Extinction': 1,
    'Oz the Great and Powerful': 1,
    'The Amazing Spider-Man 2': 1,
    'TRON: Legacy': 0,
    'Cars 2': 1,
    'Green Lantern': 0,
    'Toy Story 3': 1,
    'Terminator Salvation': 1,
    'Furious 7': 1,
    'World War Z': 1,
    'X-Men: Days of Future Past': 1,
    'Star Trek Into Darkness': 1,
    'Jack the Giant Slayer': 0,
    'The Great Gatsby': 1,
    'Prince of Persia: The Sands of Time': 0,
    'Pacific Rim': 1,
    'Transformers: Dark of the Moon': 1,
    'Indiana Jones and the Kingdom of the Crystal Skull': 1,
    'Brave': 1,
    'Star Trek Beyond': 1,
    'WALL∑E': 1,
    'Rush Hour 3': 0,
    '2012': 1,
    'Bee Movie': 0,
    'A Christmas Carol': 1,
    'Jupiter Ascending': 0,
    'The Legend of Tarzan': 0,
    'The Chronicles of Narnia: The Lion, the Witch and the Wardrobe': 1,
    'X-Men: Apocalypse': 0,
    'The Dark Knight': 1,
    'Up': 1,
    'Monsters vs. Aliens': 1,
    'Iron Man': 1,
    'Hugo': 1,
    'Wild Wild West': 0,
    'The Mummy: Tomb of the Dragon Emperor': 1,
    'Suicide Squad': 0,
    'Evan Almighty': 1,
    'Edge of Tomorrow': 1,
    'Waterworld': 0,
    'G.I. Joe: The Rise of Cobra': 0,
    'Inside Out': 1,
    'The Jungle Book': 1,
    'Iron Man 2': 1,
    'Snow White and the Huntsman': 1,
    'Maleficent': 1,
    'Dawn of the Planet of the Apes': 1
}

# convert movie_dataset into pd.dataframe
movie_dataset_df = pd.DataFrame(movie_dataset)
# transpose df so that movie name is rows and features is columns
movie_dataset_df_transposed = movie_dataset_df.transpose()

# using sklearn to split the data
# use only movie dataset because labels has to match the movies chosen for training and validation
t_set, v_set = (
    train_test_split(movie_dataset_df_transposed, test_size=0.2, random_state=42))


# change the training_set and validation_set into a dict where the key is a string of movie name
# and the value is a list of features (budget, runtime, year - all normalized)
def change_df_to_dict(df_given):
    df_dict = df_given.to_dict('index')
    final_dict = {}
    for k, v in df_dict.items():
        final_dict_values = []
        for inner_k, inner_v in v.items():
            final_dict_values.append(inner_v)
        final_dict[k] = final_dict_values
    return final_dict


training_set = change_df_to_dict(t_set)
validation_set = change_df_to_dict(v_set)
print(validation_set)


# get the labels from movie_labels of each movie in the set given (training_set, validation_set)
def get_labels(set_given):
    final_labels = {}
    for k, v in set_given.items():
        for k2, v2 in movie_labels.items():
            if k == k2:
                final_labels[k] = v2
    return final_labels


training_labels = get_labels(training_set)
validation_labels = get_labels(validation_set)
