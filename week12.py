import pandas as pd
from textblob import TextBlob
import matplotlib.pyplot as plt
import re
def clean_tweet(tweet):
    return ' '.join(re.sub(r'(@[A-Za-z0-9]+)|([^0-9A-Za-z\s])|(\w+:\/\/\S+)',' ',tweet).split())
def analyze_sentiment(tweet):
    '''Classify the sentiment polarity of a tweet using TextBlob.'''
    if isinstance(tweet,str):
        analysis=TextBlob(clean_tweet(tweet))
        if analysis.sentiment.polarity>0:
            return 'Positive'
        elif analysis.sentiment.polarity==0:
            return 'Neutral'
        else:
             return 'Negetive'
    else:
         return 'Neutral'
def perform_sentiment_analysis_on_dataset(file_path):
    try:
        data = pd.read_csv(file_path,encoding='ISO-8859-1')

    except Exception as e:
        print(f"error reading the file: (0)")
        return
    if 'tweet' not in data.colomns:
        print("The dataset does not contain 'tweet' column.Please check the column name.")
        return
    data['Sentiment']= daat['tweet'].apply(analyse_sentiment)
    print(f"Sentiment analysis result for the dataset:")
    print(f"Positive tweets: {len(data[data['Sentiment']=='Positive'])}")
    print(f"Neutral tweets: {len(data[data['Sentiment']=='Neutral'])}")
    print(f"Negetive tweets: {len(data[data['Sentiment']=='Negetive'])}")
    sentiment_counts = data['Sentiment'].value_counts()
    sentiment_labels = sentiment_counts.index
    sentiment_sizes = sentiment_counts.values
    plt.figure(figsize=(7,7))
    plt.pie(sentiment_sizes, labels,autopact='%1.1f%%',startangle=140,colors=['#4CAF50','#FFC107','#F44336'])
    plt.title("Sentiment Distribution of tweets")
    plt.axis('equal')
    plt.show()
    data.to.save('sentiment_analysis_results.csv',index=false)
    print("\nSentiment analysis results haver been saved to 'sentiment_analysis_results.csv")
    print("nSample results(Tweets with Sentiments):")
    print(data[['tweet','Sentiments']].head())
if __name__=="__main__":
        file_path = input("Enter the path to the Twitter dataset (CSV): ")
        perform_sentiment_analysis_on_dataset(file_path)
                                 
        
        
