# ML-Modelling-App
This application is a Streamlit dashboard that can be used to analyse the performance of ML models.

### Use Case: [Spotify Recommendations](https://www.kaggle.com/datasets/bricevergnou/spotify-recommendation/data)
Can you teach Spotify the kind of music to suggest to you? Take this dataset as an example. It was interesting to find that several variables defining the music such as key, danceability, energy, acousticness, etc. can be retrieved from [Spotify's API](https://developer.spotify.com/documentation/web-api). After creating the target column by categorising songs into liked and disliked, our dataset is ready to be worked upon. 

### Overview of the App
For my application, I have used Random Forest Regressor to approach the Regression problems, and Random Forest Classifier to approach the Classification problems.
When you launch the application, the visualisations for the use case get displayed. When you click on the 'Train' button at the end of the 'Train a Model' tab, you can view the performance of the model through metric scores and relevant plots. You can tune the parameters and note the change in the model's accuracy. This can also be useful to understand how different parameters affect the model.
You can even upload your own dataset and note the visualisations, and my model's performance on it.
Furthermore, if you wish to upload your own model, you can do so by clicking on the 'Upload a Model' option. You will be required to upload the relevant dataset and follow the instructions mentioned. Once you have completed the uploads, you can view the data's visualisations and your model's performance.

In case you ever run into problems, ensure that you have uploaded the correct files and have chosen the right problem type.
