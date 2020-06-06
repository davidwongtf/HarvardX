#####################################################
#                                                   #
# Author: David Wong                                #
# Course: HarvardX: PH125.9x Data Science: Capstone #
# Course provider: edX                              #
# Assignment: MovieLens Project                     #
# When: June 2020                                   #
#                                                   #
#####################################################

####################
#                  #
# Data Preparation #
#                  #
####################

# Data preparation codes provided by edx to create edx and validation data sets

# Note: this process could take a couple of minutes

if(!require(tidyverse)) install.packages("tidyverse", repos = "http://cran.us.r-project.org")
if(!require(caret)) install.packages("caret", repos = "http://cran.us.r-project.org")
if(!require(data.table)) install.packages("data.table", repos = "http://cran.us.r-project.org")

library(data.table)
library(tidyverse)
library(dplyr)
library(caret)

# MovieLens 10M dataset:
# https://grouplens.org/datasets/movielens/10m/
# http://files.grouplens.org/datasets/movielens/ml-10m.zip

dl <- tempfile()
download.file("http://files.grouplens.org/datasets/movielens/ml-10m.zip", dl)

ratings <- fread(text = gsub("::", "\t", readLines(unzip(dl, "ml-10M100K/ratings.dat"))),
                 col.names = c("userId", "movieId", "rating", "timestamp"))

movies <- str_split_fixed(readLines(unzip(dl, "ml-10M100K/movies.dat")), "\\::", 3)
colnames(movies) <- c("movieId", "title", "genres")
# Modified to resolve "NA" issues due to R 4.0
movies <- as.data.frame(movies) %>% mutate(movieId = as.numeric(movieId),
                                           title = as.character(title),
                                           genres = as.character(genres))

movielens <- left_join(ratings, movies, by = "movieId")

# Validation set will be 10% of MovieLens data
set.seed(1, sample.kind="Rounding")
# if using R 3.5 or earlier, use `set.seed(1)` instead
test_index <- createDataPartition(y = movielens$rating, times = 1, p = 0.1, list = FALSE)
edx <- movielens[-test_index,]
temp <- movielens[test_index,]

# Make sure userId and movieId in validation set are also in edx set
validation <- temp %>% 
  semi_join(edx, by = "movieId") %>%
  semi_join(edx, by = "userId")

# Add rows removed from validation set back into edx set
removed <- anti_join(temp, validation)
edx <- rbind(edx, removed)

rm(dl, ratings, movies, test_index, temp, movielens, removed)

# Split edx data set further into train_set and test_set
test_index_edx <- createDataPartition(y = edx$rating, times = 1, p = 0.2,
                                      list = FALSE)
train_set <- edx[-test_index_edx,]
test_set <- edx[test_index_edx,]

test_set <- test_set %>%
  semi_join(train_set, by = "movieId") %>%
  semi_join(train_set, by = "userId")

#####################################################################
#                                                                   #
# Data Exploration Round 1 - pave the way for movie and user effect #
#                                                                   #
#####################################################################

# Quick look to confirm the data for all data sets look clean and have the expected record counts
edx %>% as_tibble()
validation %>% as_tibble()
train_set %>% as_tibble()
test_set %>% as_tibble()

# Determine number of unique users and movies and confirm that not all users have rated all movies
unique_user <- n_distinct(edx$userId)
unique_movie <- n_distinct(edx$movieId)
unique_user * unique_movie

# Observe ratings distribution - confirm some movies rated higher than others
hist(edx$rating)

# Plot number of ratings per movie to confirm that some popular movies get rated more than others
# Pave the way for movie bias

edx %>% 
  count(movieId) %>% 
  ggplot(aes(n)) + 
  geom_histogram(bins = 30, color = "black") + 
  scale_x_log10() + 
  ggtitle("Movies")

# Some examples on movies that are rated only once
# Again, pave the way for movie bias
edx %>%
  group_by(movieId) %>%
  summarize(count = n()) %>%
  filter(count == 1) %>%
  left_join(edx, by = "movieId") %>%
  group_by(title) %>%
  summarize(rating = rating, n_rating = count) %>%
  slice(1:50) %>%
  knitr::kable()

# Plot number of ratings per user to confirm that some users are more active
# Pave the way for user bias

edx %>%
  count(userId) %>% 
  ggplot(aes(n)) + 
  geom_histogram(bins = 30, color = "black") + 
  scale_x_log10() +
  ggtitle("Users")

# Plot mean movie ratings given by users who have rated more than 100 movies
# The distribution of this particular selection is different from the general distribution
# Pave the way for user bias
edx %>%
  group_by(userId) %>%
  filter(n() >= 100) %>%
  summarize(b_u = mean(rating)) %>%
  ggplot(aes(b_u)) +
  geom_histogram(bins = 30, color = "black") +
  xlab("Mean rating") +
  ylab("Number of users") +
  ggtitle("Mean movie ratings given by users") +
  scale_x_discrete(limits = c(seq(0.5,5,0.5))) +
  theme_light()

#########################################################################
#                                                                       #
# Build and Test Models Round 1 - simple average, movie and user effect #
#                                                                       #
#########################################################################

# Model 1: Average movie rating

# Compute the mean
mu <- mean(train_set$rating)

# Test results based on simple prediction
naive_rmse <- RMSE(test_set$rating, mu)

# Create a data frame and Save result of the first model
rmse_results <- data_frame(method = "Average movie rating model (test_set)", RMSE = naive_rmse)
rmse_results %>% knitr::kable()

# Model 2: Movie effect model

# Enhance Model 1 by adding the movie bias b_i which represents average ranking for movie i
movie_avgs <- train_set %>%
  group_by(movieId) %>%
  summarize(b_i = mean(rating - mu))

# Test and append result to the existing result data frame
predicted_ratings <- mu +  test_set %>%
  left_join(movie_avgs, by='movieId') %>%
  pull(b_i)

movie_effect_rmse <- RMSE(predicted_ratings, test_set$rating)

rmse_results <- bind_rows(rmse_results,
                          data_frame(method="Movie effect model (test_set)",  
                                     RMSE = movie_effect_rmse ))

# Check results and confirm there is an improvement using movie bias
rmse_results %>% knitr::kable()

# Model 3: Movie and user effect model

# Enhance Model 2 by adding the user bias b_u
user_avgs <- train_set %>%
  left_join(movie_avgs, by='movieId') %>%
  group_by(userId) %>%
  summarize(b_u = mean(rating - mu - b_i))


# Test and append result to the existing result data frame
predicted_ratings <- test_set%>%
  left_join(movie_avgs, by='movieId') %>%
  left_join(user_avgs, by='userId') %>%
  mutate(pred = mu + b_i + b_u) %>%
  pull(pred)

user_effect_rmse <- RMSE(predicted_ratings, test_set$rating)

rmse_results <- bind_rows(rmse_results,
                          data_frame(method="Movie and user effect model (test_set)",  
                                     RMSE = user_effect_rmse))

# Check results and confirm there is an improvement using user bias on top of movie bias
rmse_results %>% knitr::kable()

#################################################################################
#                                                                               #
# Data Exploration Round 2 - pave the way for Regularized movie and user effect #
#                                                                               #
#################################################################################

# Create a list of Movie IDs and Titles
movie_titles <- edx %>%
  select(movieId, title) %>%
  distinct()

# Find out the 10 best movies according to our estimate - obscure movies!
movie_avgs %>% left_join(movie_titles, by="movieId") %>%
  arrange(desc(b_i)) %>%
  slice(1:10) %>%
  pull(title)

# Find out the 10 worst movies according to our estimate - also obscure movies!
movie_avgs %>% left_join(movie_titles, by="movieId") %>%
  arrange(b_i) %>%
  slice(1:10) %>%
  pull(title)

# Find out how often these top (best and worst) movies are rated
train_set %>% count(movieId) %>%
  left_join(movie_avgs, by="movieId") %>%
  left_join(movie_titles, by="movieId") %>%
  arrange(desc(b_i)) %>%
  slice(1:10) %>%
  pull(n)

train_set %>% count(movieId) %>%
  left_join(movie_avgs) %>%
  left_join(movie_titles, by="movieId") %>%
  arrange(b_i) %>%
  slice(1:10) %>%
  pull(n)

# Key finding: those movies are rated by a few users, on obscure movies, hence large estimates of b_i
# Need regularization to penalize large estimates formed using small sample sizes.

#####################################################################
#                                                                   #
# Build and Test Models Round 2 - Regularized movie and user effect #
#                                                                   #
#####################################################################

# Model 4: Regularized movie and user effect model

# The penalized estimate, lambda, is a tuning parameter, therefore, use cross-validation to choose it
lambdas <- seq(0, 2, 0.10)

# For each lambda,find b_i & b_u, followed by rating prediction & testing

###########################
#                         #
# Use train_set data only #
#                         #
###########################

rmses <- sapply(lambdas, function(l){
  
  mu <- mean(train_set$rating)
  
  b_i <- train_set %>% 
    group_by(movieId) %>%
    summarize(b_i = sum(rating - mu)/(n()+l))
  
  b_u <- train_set %>% 
    left_join(b_i, by="movieId") %>%
    group_by(userId) %>%
    summarize(b_u = sum(rating - b_i - mu)/(n()+l))
    
  predicted_ratings <- 
    train_set %>% 
    left_join(b_i, by = "movieId") %>%
    left_join(b_u, by = "userId") %>%
    mutate(pred = mu + b_i + b_u) %>%
    pull(pred)
  
  return(RMSE(predicted_ratings, train_set$rating))
})

# Plot rmses vs lambdas to select the optimal lambda                                                             
qplot(lambdas, rmses)  


# Determine the optimal lambda which gives the smallest RMSE value                                                             
lambda <- lambdas[which.min(rmses)]

# Test and append result to the existing result data frame
rmse_results <- bind_rows(rmse_results,
                          data_frame(method="Regularized movie and user effect model (train_set)",  
                                     RMSE = min(rmses)))

# Check results and confirm there is an improvement and also meets the target
rmse_results %>% knitr::kable()

#########################################
#                                       #
# Use test_set data with optimal lambda #
#                                       #
#########################################

mu <- mean(test_set$rating)

# Compute regularized movie bias using optimal lambda
b_i <- test_set %>% 
  group_by(movieId) %>%
  summarize(b_i = sum(rating - mu)/(n()+lambda))

# Compute regularized user bias using optimal lambda
b_u <- test_set %>% 
  left_join(b_i, by="movieId") %>%
  group_by(userId) %>%
  summarize(b_u = sum(rating - b_i - mu)/(n()+lambda))

# Test and append result to the existing result data frame
predicted_ratings <- 
  test_set %>% 
  left_join(b_i, by = "movieId") %>%
  left_join(b_u, by = "userId") %>%
  mutate(pred = mu + b_i + b_u) %>%
  pull(pred)

test_set_result <- RMSE(predicted_ratings, test_set$rating)

rmse_results <- bind_rows(rmse_results,
                          data_frame(method="Regularized movie and user effect model (test_set)",  
                                     RMSE = test_set_result))

# Check results and confirm there is an improvement on test_set and also meets the target
rmse_results %>% knitr::kable()

###################################################################################
#                                                                                 #
# Final test using Regularized Movie and User Effect model on Validation data set #
#                                                                                 #
###################################################################################

mu <- mean(validation$rating)

# Compute regularized movie bias using optimal lambda
b_i <- validation %>% 
  group_by(movieId) %>%
  summarize(b_i = sum(rating - mu)/(n()+lambda))

# Compute regularized user bias using optimal lambda
b_u <- validation %>% 
  left_join(b_i, by="movieId") %>%
  group_by(userId) %>%
  summarize(b_u = sum(rating - b_i - mu)/(n()+lambda))

# Test and append result to the existing result data frame
predicted_ratings <- 
  validation %>% 
  left_join(b_i, by = "movieId") %>%
  left_join(b_u, by = "userId") %>%
  mutate(pred = mu + b_i + b_u) %>%
  pull(pred)

final_va_result <- RMSE(predicted_ratings, validation$rating)

rmse_results <- bind_rows(rmse_results,
                            data_frame(method="Final: Regularized movie and user effect model on Validation data set",  
                                       RMSE = final_va_result))

# Check results and confirm project target is met!
rmse_results %>% knitr::kable()

#################
#               #
# End of script #
#               #
#################

