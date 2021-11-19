# Authors: Piotr Michałek s19333 & Kibort Jan s19916
# run from terminal: python collaborative_filtering --user "Piotr Michalek"

import argparse
import json
import numpy as np

from compute_scores import euclidean_score

def build_arg_parser():
    parser = argparse.ArgumentParser(description='Find users who are similar to the input user')
    parser.add_argument('--user', dest='user', required=True,
            help='Input user')
    return parser

# Finds users in the dataset that are similar to the input user
def find_similar_users(dataset, user, num_users):
    if user not in dataset:
        raise TypeError('Cannot find ' + user + ' in the dataset')

    # Compute Pearson score between input user
    # and all the users in the dataset
    scores = np.array([[x, euclidean_score(dataset, user,
            x)] for x in dataset if x != user])

    # Sort the scores in decreasing order
    scores_sorted = np.argsort(scores[:, 1])[::-1]

    # Extract the top 'num_users' scores
    top_users = scores_sorted[:num_users]

    return scores[top_users]

def add_similar_users_to_list(database, similars):
    #get similar users
    for item in database:
        for sim_user in similars:
            if(item == sim_user[0]):
                similars_users_names.append(item)

def create_movies_set():
    #get all movies by users
    for name in similars_users_names:
        # print(data[name])
        savemovies(data[name])

    #remove already seen movies
    for paul_movie in data[user]:
        movies.remove(paul_movie)
    #print(movies)

def find_recommended_movies():
    #initialize variables for describing recommended films
    score_index = 10

    #going from the top rating, we are looping through all the films
    #we check if users watched shared films and if so, we check their ratings,
    #if it's current searched top rating, we add it to recommended films
    while score_index > 0:
        for movie in movies:
            for name in similars_users_names:
                  if (len(recommended_movies) < 5):
                      if (movie in data[name] and data[name][movie] == score_index):
                          # print(movie, 'equals', data[name][movie])
                          recommended_movies.add(movie)
        score_index-=1

def find_not_recommended_movies():
    #initialize variables for describing recommended films
    score_index = 1

    #going from the top rating, we are looping through all the films
    #we check if users watched shared films and if so, we check their ratings,
    #if it's current searched top rating, we add it to recommended films
    while score_index < 11:
        for movie in movies:
            for name in similars_users_names:
                  if (len(not_recommended_movies) < 5):
                      if (movie in data[name] and data[name][movie] == score_index):
                          # print(movie, 'equals', data[name][movie])
                          not_recommended_movies.add(movie)
        score_index+=1

#get set of movies
def savemovies(user_data):
    for movie in user_data:
        movies.add(movie)

if __name__=='__main__':
    recommended_movies = set()
    not_recommended_movies = set()
    similars_users_names = []
    #we dont want repeated movies
    movies = set()
    args = build_arg_parser().parse_args()
    user = args.user

    ratings_file = 'data.json'

    with open(ratings_file, 'r') as f:
        data = json.loads(f.read())

    savemovies(data[user])

    print('\nUsers similar to ' + user + ':\n')
    similar_users = find_similar_users(data, user, 3)


    print('User\t\t\tSimilarity score')
    print('-'*41)
    for item in similar_users:
        print(item[0], '\t\t', round(float(item[1]), 2))

    add_similar_users_to_list(data, similar_users)
    create_movies_set()
    find_recommended_movies()
    find_not_recommended_movies()

    print('RECOMMENDED MOVIES', recommended_movies)
    print('NOT RECOMMENDED MOVIES: ', not_recommended_movies)

