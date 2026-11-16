#!/usr/bin/env python
# coding: utf-8

# In[22]:


import pandas as pd
import ast


# In[23]:


movies = pd.read_csv("tmdb_5000_movies.csv")
credits = pd.read_csv("tmdb_5000_credits.csv")

movies = movies.merge(credits, left_on="id", right_on="movie_id")

movies.rename(columns={"title_x": "title"}, inplace=True)

print(movies.shape)


# In[24]:


#extract movie information
def get_names(text):
    return [item["name"] for item in ast.literal_eval(text)]

movies["genres"] = movies["genres"].apply(get_names)
movies["keywords"] = movies["keywords"].apply(get_names)

movies["cast"] = movies["cast"].apply(lambda x: get_names(x)[:3])

movies["crew"] = movies["crew"].apply(
    lambda x: [item["name"] for item in ast.literal_eval(x)
               if item["job"] == "Director"][:1]
)


# In[25]:


#movie tags
movies["overview"] = movies["overview"].fillna("")

movies["tags"] = (
    movies["overview"] + " " +
    movies["genres"].apply(lambda x: " ".join(x)) + " " +
    movies["keywords"].apply(lambda x: " ".join(x)) + " " +
    movies["cast"].apply(lambda x: " ".join(x)) + " " +
    movies["crew"].apply(lambda x: " ".join(x))
)


# In[27]:


new_movies = movies[["id", "title", "genres", "overview", "tags"]].copy()

new_movies.to_csv("movies_processed.csv", index=False)

print("Movie data processed successfully!")


# In[28]:


new_movies[["title", "tags"]].head()


# In[31]:


#TF-IDF: Converts movie descriptions into numerical values.
#Cosine Similarity: Finds movies that are similar to the selected movie.

from sklearn.feature_extraction.text import TfidfVectorizer

tfidf = TfidfVectorizer(stop_words="english", max_features=5000)

tfidf_matrix = tfidf.fit_transform(new_movies["tags"])

print("TF-IDF completed!")
print("Matrix shape:", tfidf_matrix.shape)


# In[32]:


from sklearn.metrics.pairwise import cosine_similarity

similarity = cosine_similarity(tfidf_matrix)

print("Similarity calculated!")
print("Matrix shape:", similarity.shape)


# In[33]:


def recommend(movie_name):

    movie_index = new_movies[
        new_movies["title"].str.lower() == movie_name.lower()
    ].index[0]

    scores = list(enumerate(similarity[movie_index]))

    scores = sorted(scores, key=lambda x: x[1], reverse=True)

    for i in scores[1:6]:
        print(new_movies.iloc[i[0]]["title"])


# In[34]:


recommend("Avatar")


# In[ ]:




