This project involves developing a Python-based Streamlit app to analyze comments extracted from a 
web page using Web Scraping techniques. The primary focus is on text classification tasks, and it 
utilizes various models from the HuggingFace library. 
The aim is to classify comments into different categories using these pre-trained models.

The app takes a URL as input, scrapes the comments from the provided web page, and applies HuggingFace's 
models to classify the comments. The models used include the sentiment analysis model from 
"nlptown/bert-base-multilingual-uncased-sentiment" and additional models for identifying racism and 
political content.

By using Streamlit, the project enables a user-friendly interface for users to interact with the app. 
After analyzing the comments, the app displays the average sentiment score, the proportion of comments 
classified as racism or political, and presents each comment that falls into these categories.

Overall, the project aims to provide an efficient and user-friendly tool to analyze comments from web pages, 
assisting users in understanding the sentiment and potential presence of racism or political content within 
the comments.