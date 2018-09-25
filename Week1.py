"""
Sept 19
First team activity
To be completed in class with group
"""

import random
import numpy as np


class Movie:
    def __init__(self, title = "", year = 0, runtime = 0):
        self.title = title
        self.year = year
        if runtime < 0:
            self.runtime = 0
        else:
            self.runtime = runtime
        
    def __repr__(self):
        return "%s (%d) - %d mins." % (self.title, self.year, self.runtime)
    
    def convert_runtime(self):
        hours = self.runtime // 60
        minutes = self.runtime % 60
        
        return hours, minutes
    

# 2.1
def create_movie_list():
    movie_list = []
    movie_list.append(Movie("Star Wars", 1994, 160))
    movie_list.append(Movie("Lord of the Rings", 1995, 210))
    movie_list.append(Movie("Hercules", 1996, 130))
    movie_list.append(Movie("Narnia", 1997, 134))
    
    return movie_list
    
    
# 3.1
def get_movie_data():
    """
    Generate a numpy array of movie data
    :return:
    """
    num_movies = 10
    array = np.zeros([num_movies, 3], dtype=np.float)

    #random = Random()

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


def main():
    m = Movie("Harry Potter", 2001, 152)
    
    print(m)
    
    print("%d hours %d minutes" % (m.convert_runtime()))
    
    # 2.2
    movies = create_movie_list()
    
    for movie in movies:
        print(movie)
        
    # 2.3
    # Create a list of movies with runtime > 150
    long_movies = [m for m in movies if m.runtime > 150]
    
    for movie in long_movies:
        print(movie)
        
    # 2.4
    # Dictionary
    # Add random stars (0 <= star <= 5) to every title in list
    stars_map = {m.title : random.uniform(0, 5) for m in movies}
    
    # 2.5
    # Display each title and their stars
    # How does stars_map[title] get the star?
    for title in stars_map:
        print("{} - {:.2f} Stars".format(title, stars_map[title]))
        
    # 3.2
    mydata = get_movie_data()
    
    rows = mydata.shape[0]
    cols = mydata.shape[1]
    
    print("{} rows \n{} columns".format(rows, cols))
    
    # 3.3
    print(mydata[0:2])
    
    # 3.4
    print(mydata[:, -2])
    
    # 3.5
    print(mydata[:,2])
    


if __name__ == "__main__":
    main()
    
    
 
