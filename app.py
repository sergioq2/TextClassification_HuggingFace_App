import requests
from bs4 import BeautifulSoup
import re
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import pandas as pd
import numpy as np
import torch
import matplotlib.pyplot as plt
import streamlit as st

#verify if the user has a GPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

#load the models with the tokenizer
tokenizer_sent = AutoTokenizer.from_pretrained("nlptown/bert-base-multilingual-uncased-sentiment")
model_sent = AutoModelForSequenceClassification.from_pretrained("nlptown/bert-base-multilingual-uncased-sentiment")

tokenizer_racism = AutoTokenizer.from_pretrained("davidmasip/racism")
model_racism = AutoModelForSequenceClassification.from_pretrained("davidmasip/racism")

tokenizer_politic = AutoTokenizer.from_pretrained("Newtral/xlm-r-finetuned-toxic-political-tweets-es")
model_politic = AutoModelForSequenceClassification.from_pretrained("Newtral/xlm-r-finetuned-toxic-political-tweets-es")

#function to get the reviews from the url
def request_web(url):
    r = requests.get(url)
    soup = BeautifulSoup(r.text, 'html.parser')
    regex = re.compile('.*comment.*')
    results = soup.find_all('p', {'class': regex})
    reviews = [result.text for result in results]
    return reviews

#function to create a dataframe with the reviews
def dataframe(reviews):
    df = pd.DataFrame(np.array(reviews), columns=['review'])
    return df

#function to get the sentiment score of a review
def sentiment_score(review):
    tokens = tokenizer_sent.encode(review, return_tensors='pt', truncation=True, max_length=512)
    result = model_sent(tokens)
    return int(torch.argmax(result.logits)) + 1

#function to get the racism score of a review
def racism_score(review):
    tokens = tokenizer_racism.encode(review, return_tensors='pt', truncation=True, max_length=512)
    result = model_racism(tokens)
    return int(torch.argmax(result.logits))

#function to get the political score of a review
def politic_score(review):
    tokens = tokenizer_politic.encode(review, return_tensors='pt', truncation=True, max_length=512)
    result = model_politic(tokens)
    return int(torch.argmax(result.logits))

#function to get the average sentiment score of the reviews
def average(df):
    df['Sentiment'] = df['review'].apply(lambda x: sentiment_score(x))
    df['Racism'] = df['review'].apply(lambda x: racism_score(x))
    df['Politic'] = df['review'].apply(lambda x: politic_score(x))
    racism_proportion = df['Racism'].sum() / len(df['Racism'])
    politic_proportion = df['Politic'].sum() / len(df['Politic'])
    average = df['Sentiment'].mean()
    racism_comments = df[df['Racism'] == 1]['review'].tolist()
    politic_comments = df[df['Politic'] == 1]['review'].tolist()
    sad_comments = df[df['Sentiment'] == 1]['review'].tolist()
    return average, racism_comments, politic_comments, sad_comments, racism_proportion, politic_proportion

#main function
def main():
    st.title("Analysis of Sentiment, Racism and Political Comments")
    st.write("Enter a URL to scrape reviews and get the results!")

    url = st.text_input("Enter the URL:", "")

    if url:
        try:
            reviews = request_web(url)
            df = dataframe(reviews)
            average_score, racism_comments, politic_comments, sad_comments, racism_proportion, politic_proportion = average(df)
            
            plot_sentiment = df['Sentiment'].value_counts().plot(kind='bar')
            st.pyplot(plot_sentiment.figure)

            st.write("Average Sentiment Score (from 1 to 5): {:.2f}".format(average_score))
            st.subheader("Sad Comments:")
            for comment in sad_comments:
                st.write("- " + comment)

            plot_racism = df['Racism'].value_counts().plot(kind='pie')
            st.pyplot(plot_racism.figure)

            st.write("Proportion of racism comments: {:.2f}".format(racism_proportion))
            st.subheader("Racism Comments:")
            for comment in racism_comments:
                st.write("- " + comment)

            plot_politic = df['Racism'].value_counts().plot(kind='barh')
            st.pyplot(plot_politic.figure)
            
            st.write("Proportion of political comments: {:.2f}".format(politic_proportion))
            st.subheader("Political Comments:")
            for comment in politic_comments:
                st.write("- " + comment)
        
        
            
        except Exception as e:
            st.error(f"Error occurred: {e}")
            st.write("Make sure the provided URL is valid and contains review elements.")

if __name__ == "__main__":
    main()
