import pyspark
from pyspark import SparkConf, SparkContext
import collections

## Create a SparkConf object, which configures an application and sets parameters
conf = SparkConf().setAppName("MovieAnalysis")
## Create a SparkContext object, which is the connection to a Spark cluster
sc = SparkContext(conf = conf)

## Load the data
ratingLines = sc.textFile("hdfs://cluster-6902-m/wsu/ratings.csv")
movieLines = sc.textFile("hdfs://cluster-6902-m/wsu/movies.csv")

## View the data
#ratingLines.take(5)
#movieLines.take(12)

## First, clean movie lines data
## Create the header row
Movie_Header = movieLines.zipWithIndex() \
	.filter(lambda index: index[1] == 0) \
	.map(lambda x: x[0])
Movie_SplitHeader = Movie_Header.map(lambda x: x.split(","))
## Take the data without the header row
Movie_NoHeader = movieLines.zipWithIndex() \
	.filter(lambda index: index[1] > 0) \
	.map(lambda x: x[0])
## if the data does contain inline quotes
ContainsQuotes = Movie_NoHeader.filter(lambda x: "\"" in x.encode('utf-8'))
SplitQuotedString = ContainsQuotes.map(lambda x: x.replace(",\"", "\",")) \
	.map(lambda x: x.split("\","))
## if the data does not contain inline quotes
NoQuotes = Movie_NoHeader.filter(lambda x: "\"" not in x.encode('utf-8'))
SplitUnquotedString = NoQuotes.map(lambda x: x.split(","))
## Bring it back together
Movie_Split = SplitQuotedString.union(SplitUnquotedString)

## Change the ID to an INT. Now in the form MovieID, title, genres
Movies = Movie_Split.map(lambda x: [str(x[0]), x[1], x[2]])

## Next, clean the rating lines data
## Create the header row
Rating_Header = ratingLines.zipWithIndex() \
	.filter(lambda index: index[1] == 0) \
	.map(lambda x: x[0])
Rating_SplitHeader = Rating_Header.map(lambda x: x.split(","))
## Take the data without the header row
Rating_NoHeader = ratingLines.zipWithIndex() \
	.filter(lambda index: index[1] > 0) \
	.map(lambda x: x[0])
Rating_Split = Rating_NoHeader.map(lambda x: x.split(","))
## Change data types: now in the form UserID, MovieID, Rating, TimeStamp
Ratings = Rating_Split.map(lambda x: [str(x[0]), str(x[1]), float(x[2]), int(x[3])])



### Movie Stats:

## All movies and ratings
movies_and_ratings = Movies.map(lambda x: (x[0], x[1])) \
	.join(Ratings.map(lambda x: (x[1], x[2]))) \
	.map(lambda x: x[1])

## Average rating for each movie
movies_rated = sc.parallelize(list(movies_and_ratings.collect())) \
	.mapValues(lambda v: (v, 1)) \
    .reduceByKey(lambda a,b: (a[0]+b[0], a[1]+b[1])) \
    .mapValues(lambda v: v[0]/v[1])

## Number of ratings per movie
movie_rating_counts = sc.parallelize(list(movies_and_ratings.collect())) \
	.mapValues(lambda v: (v, 1)) \
    .reduceByKey(lambda a,b: (a[0]+b[0], a[1]+b[1])) \
    .mapValues(lambda v: v[1])

## Top and Bottom overall rating
TopRating = movies_rated.map(lambda x: x[1]) \
	.max()

BottomRating = movies_rated.map(lambda x: x[1]) \
	.min()

## List of all movies with highest and lowest ratings
TopMovies = movies_rated.filter(lambda x: x[1] == TopRating) \
	.map(lambda x: x[0]) \
	.collect()

BottomMovies = movies_rated.filter(lambda x: x[1] == BottomRating) \
	.map(lambda x: x[0]) \
	.collect()

## Min count of ratings for top and bottom movies where rating in the top 5
Top5Movies_MinRatingCount = sc.parallelize(movie_rating_counts.filter(lambda x: x[0] in TopMovies) \
	.top(5, key=lambda x: x[1])) \
	.map(lambda x: x[1]) \
	.min()

Bottom5Movies_MinRatingCount = sc.parallelize(movie_rating_counts.filter(lambda x: x[0] in BottomMovies) \
	.top(5, key=lambda x: x[1])) \
	.map(lambda x: x[1]) \
	.min()

## Pull all movies that are in the top list and have enough ratings to make the top 5. May show more than 5 if ties
TopMovieslist = movie_rating_counts.filter(lambda x: x[0] in TopMovies) \
	.filter(lambda x: x[1] >= Top5Movies_MinRatingCount) \
	.map(lambda x: x[0]) \
	.collect()

BottomMovieslist = movie_rating_counts.filter(lambda x: x[0] in BottomMovies) \
	.filter(lambda x: x[1] >= Bottom5Movies_MinRatingCount) \
	.map(lambda x: x[0]) \
	.collect()

## Get the data for printing
Top_Movies_Rating = collections.OrderedDict(movies_and_ratings.filter(lambda x: x[0] in TopMovieslist) \
	.sortBy(lambda x: x[0]) \
	.collect())
Bottom_Movies_Rating = collections.OrderedDict(movies_and_ratings.filter(lambda x: x[0] in BottomMovieslist) \
	.sortBy(lambda x: x[0]) \
	.collect())







## Output
print("\n")
print("\n")
print("Top Movies, sorted alphabetically:")
print("\n")
for key, value in Top_Movies_Rating.items():
	print("%s, rating: %i/5" % (key, value))
print("\n")
print("\n")
print("Bottom Movies, sorted alphabetically:")
print("\n")
for key, value in Bottom_Movies_Rating.items():
	print("%s, rating: %i/5" % (key, value))