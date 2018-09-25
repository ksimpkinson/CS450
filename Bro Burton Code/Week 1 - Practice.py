"""
Author: Brother Burton
Description: Python Practice for CS 450, Week 01 Practice
"""


# Part 1 - Classes
import random
import numpy as np

# 1.1
class Movie:
    """
    This class keeps track of the elements of a movie
    """
    def __init__(self, title="", year=0, runtime=0):
        self.title = title
        self.year = year

        if runtime < 0:
            runtime = 0

        self.runtime = runtime

    def __repr__(self):
        return "{} ({}) - {} mins".format(self.title, self.year, self.runtime)

    def get_runtime_hours_minutes(self):
        hours = self.runtime // 60
        minutes = self.runtime % 60

        return hours, minutes


# 1.2 Create a new movie
m = Movie()
m.title = "Jurassic World"
m.year = 2015
m.runtime = 124

# 1.3
print(m)

# 1.4
hours, minutes = m.get_runtime_hours_minutes()
print("hours: {}, minutes: {}".format(hours, minutes))

# 1.5
m2 = Movie("Beauty and the Beast", 1991, 84)
print(m2)

# 1.6
m3 = Movie("Test", 2000, -182)
print(m3)


# Part 2

# 2.1
def create_movie_list():
    movies = []
    movies.append(Movie("Jurassic World", 2015, 124))
    movies.append(Movie("Beauty and the Beast", 1991, 84))
    movies.append(Movie("Titanic", 1997, 195))
    movies.append(Movie("test4", 2001, 400))
    movies.append(Movie("test5", 2018, 50))

    return movies

def run_part2():
    # 2.2
    movies = create_movie_list()

    print("\nIn main... Displaying Movies:")
    for movie in movies:
        print(movie)

    # 2.3
    long_movies = [m for m in movies if m.runtime > 150]

    print("\nLong Movies:")
    for movie in long_movies:
        print(movie)

    # 2.4
    stars_map = {m.title : random.uniform(0, 5) for m in movies}

    # 2.5
    print ("\nNumber of Stars:")
    for title in stars_map:
        print("{} - {:.2f} Stars".format(title, stars_map[title]))

# 3.1
def get_movie_data():
    """
    Generate a numpy array of movie data
    :return:
    """
    num_movies = 10
    array = np.zeros([num_movies, 3], dtype=np.float)

    for i in range(num_movies):
        # There is nothing magic about 100 here, just didn't want ids
        # to match the row numbers
        movie_id = i + 100

        # Lets have the views range from 100-10000
        views = random.randint(100, 10000)
        stars = random.uniform(0, 5)

        array[i][0] = movie_id
        array[i][1] = views
        array[i][2] = stars

    return array

def run_part3():
    data = get_movie_data()

    print(data)
    rows = data.shape[0]
    cols = data.shape[1]

    # 3.2
    print("There are {} rows and {} cols".format(rows, cols))

    # 3.3
    print(data[0:2])

    # 3.4
    print(data[:,-2:])

    # 3.5
    print(data[:,2])

def main():
    run_part2()
    run_part3()

if __name__ == "__main__":
    main()