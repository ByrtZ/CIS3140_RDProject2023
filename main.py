from keys import *
from wordcloud import STOPWORDS

import datetime
import string
import os
import sys
import time
import requests
import uuid
import numpy as np
import wordcloud as wc
import tweepy as twt
import pandas as pd
import textblob as tb
import text2emotion as te
import regex as re
import matplotlib.pyplot as plt


def start():
    print("\n--------------------------------\n<*> Twitter Tweets Gatherer & Analyser <*>\n--------------------------------\n")
    option = input("Choose an option to begin:\n[1] - Tweets gatherer.\n[2] - Tweets analyser.\n[x] - Exit system\n: ")
    if option == "1":
        print("Tweets gathering selected, loading system...")
        time.sleep(2)
        keywordSelector()
    elif option == "2":
        print("Tweets analyser selected, loading system...")
        time.sleep(2)
        selectDataSet()
    elif option.lower() == "x":
        print("Exiting system...")
        time.sleep(1)
        sys.exit("System closed successfully.")
    else:
        print("Invalid option, please try again.")
        time.sleep(2)
        start()


def keywordSelector():
    print("\n--------------------------------\n<*> Twitter Tweets Gatherer <*>\n--------------------------------\n")
    keyword = input("\nInput keyword to search tweets from the past 7 days: ")
    keyword = str(keyword)
    if type(keyword) is str:
        print("Checking input...")
        if "," in keyword:
            keywordsSplit = keyword.split(", ")
            keywordsSplit = ' OR '.join(keywordsSplit)
            keywordsSplit = "(" + keywordsSplit + ")"
            if confirmKeywords(keywordsSplit):
                twitterRequest(keywordsSplit)
            else:
                requestCancelled()
        else:
            if confirmKeywords(keyword):
                twitterRequest(keyword)
            else:
                requestCancelled()
    else:
        failedKeywordValidation()


def confirmKeywords(keyword):
    confirmation = ""
    while confirmation not in ["y", "n"]:
        confirmation = input("Would you like to continue with the request [Y/N]?\nYour parameter(s): " + keyword + "\n").lower()
    return confirmation == "y"


def confirmMaxResult(keyword):
    activeKeyword = keyword
    max_results = input("How many tweets should be collected? Between 10 and 100:\n")
    if max_results.isdigit():
        max_results = int(max_results)
        if 10 <= max_results <= 100:
            print("Amount of results to obtain: " + str(max_results))
            time.sleep(1)
            return max_results
        else:
            print("Amount of tweets to be collected must be between 10 and 100, please try again.")
            time.sleep(2)
            twitterRequest(activeKeyword)
    else:
        print("Invalid input, please try again.")
        time.sleep(2)
        twitterRequest(activeKeyword)


def twitterRequest(keyword):
    max_results = confirmMaxResult(keyword)
    print("\nMaking request to Twitter to find results that contain '" + keyword + "'.")
    client = twt.Client(bearer_token=BEARER_TOKEN,
                        consumer_key=API_KEY,
                        consumer_secret=API_SECRET,
                        access_token=ACCESS_TOKEN,
                        access_token_secret=ACCESS_SECRET,
                        return_type=requests.Response,
                        wait_on_rate_limit=True)
    query = keyword + " -is:retweet -is:quote -is:reply lang:en"
    print("QUERY: " + query)
    tweets = client.search_recent_tweets(query=query, tweet_fields=['author_id', 'created_at'], max_results=max_results)
    print("Request complete, creating data frame.")
    createVariables(tweets, keyword)


def createVariables(tweets, keyword):
    tweets_dict = tweets.json()
    tweets_data = tweets_dict['data']
    tweetsDataFrame = pd.json_normalize(tweets_data)
    print("Data normalised and beginning file output.")
    buildOutputFile(keyword, tweetsDataFrame)


def buildOutputFile(keyword, tweetsDataFrame):
    currentDateTime = datetime.datetime.now().strftime("%Y-%m-%d-%H:%M:%S.%f")
    keyword = keyword.replace(" OR ", "-").replace(" ", "-").replace("(", "").replace(")", "")
    dataFileName = "tweets-" + keyword + "-" + str(currentDateTime) + ".csv"
    tweetsDataFrame.to_csv(dataFileName)
    print("\nFinished collecting tweets and successfully output data to " + dataFileName + ".")
    time.sleep(2)
    print("\nReturning to system start.")
    time.sleep(2)
    start()


def selectDataSet():
    print("\n--------------------------------\n<*> Tweets Dataset Analyser <*>\n--------------------------------\n")
    time.sleep(1)
    dataSets = [ds for ds in os.listdir('.') if os.path.isfile(ds) and not ds.startswith("analysed-tweets") and ds.endswith(".csv")]
    if not dataSets:
        print("No data sets found.\nGenerate at least one data set before attempting to use this feature.\nReturning to menu...\n")
        time.sleep(3)
        start()
    else:
        i = 1
        for ds in dataSets:
            print("[" + str(i) + "] " + ds)
            i += 1
        selection = input("Select a dataset for analysis using the index in the list above: ")
        if selection.isdigit():
            selection = int(selection)
            selectedDataSet = dataSets[selection - 1]
            print("Selected dataset: " + selectedDataSet)
            time.sleep(1)
            print("Loading data analysis methods...")
            time.sleep(1)
            analyseDataSet(selectedDataSet)
        else:
            print("Invalid input, please try again.")
            time.sleep(2)
            selectDataSet()


def analyseDataSet(selectedDataSet):
    print("Beginning dataset analysis for '" + selectedDataSet + "'...\n")
    print("Creating new analysis data frame.")
    analysedDataFrame = createNewAnalysedDataFrame()
    analysedFileName = createNewAnalysedFileName()
    time.sleep(2)
    print("Sentiment analysis starting...")
    time.sleep(1)
    analysis(selectedDataSet, analysedDataFrame, analysedFileName)
    time.sleep(3)
    start()


def analysis(selectedDataSet, analysedDataFrame, analysedFileName):
    # Sentiment analysis
    positiveArray = []
    neutralArray = []
    negativeArray = []

    df = pd.read_csv(selectedDataSet, index_col=0, skipinitialspace=True)

    index = 0
    allSentiments = []
    allHappy = []
    allAngry = []
    allSurprise = []
    allSad = []
    allFear = []
    totalTweets = df['text'].count()

    for tweet in df['text']:
        tweet = cleanTweet(tweet)
        blob = tb.TextBlob(tweet)
        blob.tokenize()
        if blob.sentiment.polarity < 0:  # -ve polarity
            negativeArray.append(tweet)

        elif blob.sentiment.polarity > 0:  # +ve polarity
            positiveArray.append(tweet)

        elif blob.sentiment.polarity == 0:  # neu polarity
            neutralArray.append(tweet)

        sentiment_score = blob.sentiment.polarity
        allSentiments.append(sentiment_score)

        nextEmotion = plotEmotionVector(tweet, sentiment_score, analysedDataFrame, analysedFileName, index)  # Emotion Vectors

        allHappy.append(getIndividualEmotion(nextEmotion, "Happy"))
        allAngry.append(getIndividualEmotion(nextEmotion, "Angry"))
        allSurprise.append(getIndividualEmotion(nextEmotion, "Surprise"))
        allSad.append(getIndividualEmotion(nextEmotion, "Sad"))
        allFear.append(getIndividualEmotion(nextEmotion, "Fear"))
        index += 1

    # Evaluation metrics
    averagePolarity = sum(allSentiments) / len(allSentiments)
    averageHappy = sum(allHappy) / len(allHappy)
    averageAngry = sum(allAngry) / len(allAngry)
    averageSurprise = sum(allSurprise) / len(allSurprise)
    averageSad = sum(allSad) / len(allSad)
    averageFear = sum(allFear) / len(allFear)
    dataSetRatingValue = (averagePolarity * 0.6) + ((averageHappy * 0.2 + averageAngry * 0.2 + averageSurprise * 0.2 + averageSad * 0.2 + averageFear * 0.2) * 0.4)
    dataSetRating = 0

    print("\nPositive: " + str(len(positiveArray)))
    print("Negative: " + str(len(negativeArray)))
    print("Neutral: " + str(len(neutralArray)) + "\n")
    print("Total Number of Tweets: " + str(totalTweets))
    print("Average Sentiment Polarity: " + str(averagePolarity))
    print("Average Emotion Scores:\nHappy: " + str(averageHappy) + "\nAngry: " + str(averageAngry) + "\nSurprise: " + str(averageSurprise) + "\nSad: " + str(averageSad) + "\nFear: " + str(averageFear) + "\n")

    print("Dataset Rating: " + str(dataSetRatingValue))
    if dataSetRatingValue > 0.5:
        print("5***** STAR RATING!")
        dataSetRating = 5
    elif 0.5 > dataSetRatingValue > 0:
        print("4**** STAR RATING!")
        dataSetRating = 4
    elif dataSetRatingValue == 0:
        print("3*** STAR RATING!")
        dataSetRating = 3
    elif -0.5 < dataSetRatingValue < 0:
        print("2** STAR RATING!")
        dataSetRating = 2
    elif dataSetRatingValue < -0.5:
        print("1* STAR RATING!")
        dataSetRating = 1

    print("Generating word cloud...")
    allTweets = " ".join(tweet for tweet in df['text'])
    stopwords = set(STOPWORDS)
    stopwords.update(["https", "t", "co"])
    wordCloud = wc.WordCloud(stopwords=stopwords, background_color="white").generate(allTweets)
    print("Displaying word cloud.")
    plt.imshow(wordCloud, interpolation='bilinear')
    plt.axis("off")
    plt.show()

    print("Generating graph of averages...")
    x = np.array(["Sentiment Polarity", "Happy", "Angry", "Surprise", "Sad", "Fear", "Rating Value", "Rating"])
    y = np.array([averagePolarity, averageHappy, averageAngry, averageSurprise, averageSad, averageFear, dataSetRatingValue, dataSetRating])
    print("Displaying graph.")
    plt.bar(x, y)
    plt.show()

    print("---- <*> SENTIMENT ANALYSIS COMPLETE, EMOTION VECTORS PLOTTED AND DATA OUTPUT FOR DATASET: " + selectedDataSet + " <*> ----")
    print("---- <*> ANALYSED DATA HAS BEEN SENT TO: " + analysedFileName + " <*> ----")
    time.sleep(3)


# Emotion vectors
def plotEmotionVector(tweet, sentiment_score, analysedDataFrame, analysedFileName, index):
    print("Plotting emotion vector.")
    tweetEmotions = te.get_emotion(tweet)

    # Create data object
    data = [{'text': str(tweet).strip("\n"),
             'sentiment_score': sentiment_score,
             'happy': tweetEmotions.get("Happy"),
             'angry': tweetEmotions.get("Angry"),
             'surprise': tweetEmotions.get("Surprise"),
             'sad': tweetEmotions.get("Sad"),
             'fear': tweetEmotions.get("Fear")
             }]

    print("Emotion vector complete.")
    newAnalysedDataFrame = analysedDataFrame.append(data, ignore_index=True)
    print("Appended onto the analysed data frame, writing new row to current file.")
    writeAnalysedData(newAnalysedDataFrame, analysedFileName, index)
    return tweetEmotions


# Utility functions
def requestCancelled():
    print("Request cancelled, returning to start.")
    time.sleep(3)
    start()


def failedKeywordValidation():
    print("Unable to validate input, please try again.")
    time.sleep(3)
    start()


def getIndividualEmotion(tweetEmotions, emotion):
    if emotion == "Happy":
        return tweetEmotions.get("Happy")
    elif emotion == "Angry":
        return tweetEmotions.get("Angry")
    elif emotion == "Surprise":
        return tweetEmotions.get("Surprise")
    elif emotion == "Sad":
        return tweetEmotions.get("Sad")
    elif emotion == "Fear":
        return tweetEmotions.get("Fear")
    else:
        print("Invalid emotion found " + emotion + " while attempting to get emotion score.")
        return None


def cleanTweet(tweet):
    tweet = re.sub('@[^\s]+', '', tweet)
    tweet = re.sub('http[^\s]+', '', tweet)
    tweet = re.sub('@[A-Za-z0–9_]+', '#', tweet)
    ascii_chars = set(string.printable)
    tweet = ''.join(filter(lambda x: x in ascii_chars, tweet))
    cleanedTweet = str(tweet)
    return cleanedTweet


def randomUUID():
    return str(uuid.uuid4()).replace('=', '')


def createNewAnalysedDataFrame():
    df = pd.DataFrame()
    df['text'] = ""
    df['sentiment_score'] = ""
    df['happy'] = ""
    df['angry'] = ""
    df['surprise'] = ""
    df['sad'] = ""
    df['fear'] = ""
    return df


def createNewAnalysedFileName():
    return str("analysed-tweets-" + randomUUID() + ".csv")


def writeAnalysedData(newAnalysedDataLine, analysedFileName, newFileIndex):
    if newFileIndex == 0:  # Generate header before first line
        newAnalysedDataLine.to_csv(analysedFileName, mode='a')
    else:  # Do not create a header for extra lines
        newAnalysedDataLine.to_csv(analysedFileName, mode='a', header=False)
    print("Completed tweet analysis and successfully output data to " + analysedFileName + ".")


# System start
if __name__ == '__main__':
    start()
