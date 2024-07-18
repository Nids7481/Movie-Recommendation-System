I've developed a movie recommender web application using Streamlit for the front end and machine learning techniques like feature extraction and cosine similarity for the backend. 
The app uses the movie's features (genres, keywords, tagline, cast, and director) to find and recommend 30 similar movies.
Working:
The movie dataset is loaded into a pandas DataFrame.
Selected features are combined into a single string for each movie to create a comprehensive representation.
The combined features are vectorized using TfidfVectorizer, which converts text data into numerical features 
Cosine similarity is calculated between the feature vectors to measure the similarity between movies.
Users input their favorite movie name using a text input box.
The application finds the closest match to the input movie name in the dataset using difflib.
The app retrieves the index of the matched movie and computes similarity scores for all other movies.
Movies are then sorted based on similarity scores in descending order.
The top 30 most similar movies are displayed as recommendations.
