import imdb

# Create an instance of the IMDb class
imdb_instance = imdb.IMDb()

# Search for the movie by title
title = "Toy Story"
search_results = imdb_instance.search_movie(title)

# Select the first result
movie_id = search_results[0].getID()

# Retrieve the movie details using the movie ID
movie = imdb_instance.get_movie(movie_id)

# Print the movie details
print("Title:", movie["title"])
print("Plot:", movie["plot"][0])
print("Genres:", movie["genres"])
print("Release year:", movie["year"])
print("Rating:", movie["rating"])