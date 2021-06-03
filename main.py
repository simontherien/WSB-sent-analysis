# https://github.com/cjhutto/vaderSentiment#python-demo-and-code-examples
# https://medium.com/ro-data-team-blog/nlp-how-does-nltk-vader-calculate-sentiment-6c32d0f5046b#id_token=eyJhbGciOiJSUzI1NiIsImtpZCI6IjE3MTllYjk1N2Y2OTU2YjU4MThjMTk2OGZmMTZkZmY3NzRlNzA4ZGUiLCJ0eXAiOiJKV1QifQ.eyJpc3MiOiJodHRwczovL2FjY291bnRzLmdvb2dsZS5jb20iLCJuYmYiOjE2MjI0Njc0MzIsImF1ZCI6IjIxNjI5NjAzNTgzNC1rMWs2cWUwNjBzMnRwMmEyamFtNGxqZGNtczAwc3R0Zy5hcHBzLmdvb2dsZXVzZXJjb250ZW50LmNvbSIsInN1YiI6IjExNjE0NTU3MDMyODg2ODcxMzUxMiIsImVtYWlsIjoic2ltb24udGhlcmllbkBnbWFpbC5jb20iLCJlbWFpbF92ZXJpZmllZCI6dHJ1ZSwiYXpwIjoiMjE2Mjk2MDM1ODM0LWsxazZxZTA2MHMydHAyYTJqYW00bGpkY21zMDBzdHRnLmFwcHMuZ29vZ2xldXNlcmNvbnRlbnQuY29tIiwibmFtZSI6IlNpbW9uIFRoZXJpZW4iLCJwaWN0dXJlIjoiaHR0cHM6Ly9saDMuZ29vZ2xldXNlcmNvbnRlbnQuY29tL2EtL0FPaDE0R2dNNUJ1MTVnZWMyZ2pIZGlNQm9XQlA2SUVJZlE2YkE3emFJSmQ1QWc9czk2LWMiLCJnaXZlbl9uYW1lIjoiU2ltb24iLCJmYW1pbHlfbmFtZSI6IlRoZXJpZW4iLCJpYXQiOjE2MjI0Njc3MzIsImV4cCI6MTYyMjQ3MTMzMiwianRpIjoiYzM2M2UwY2I3NTFkZTM1MjY2NTg3Yzc2MTlhNDc1MTMzMTdlNDA4OSJ9.RXROeWiqhiVxLO_Y14wBxEVq7tF_VJj22paCrMXEqjXFRgsVr5iDfZvPLEgyzKKrVjVLWddsNmupsI212enWU3Lz7UCds2DksfCvXzzNWgsAaN3lC1Hh1QxBjGEQCMx2-ci0G5TYnzjfU6tiZXPGeg8cUdkhUbEB7hf7ClcuBmdTpKHpkxz-hnBEcnSf_Af7Q9rYbgdBEc3Q6WUZSLiQ6jcrfW_qN2hCWHaCiTSicC9RwK7XlM_Jj__9bpMOEoqguu4jF_C7wFO0Jju1sfXMRXcrlHpQxui3uh5sEkDcSGgnlxDG4XcTgy7ev75TCgkIYCS8c5kszhOJaq-b-o1rqg

"""
Momentum based strategy

Scraped WSB sentiment, got the top + most positively mentioned stocks on WSB. Strategy rebalancing monthly.
What it uses is VADER (Valence Aware Dictionary for Sentiment Reasoning), which is a model used for text sentiment
analysis that is sensitive to both polarity (positive/negative) and intensity (strength) of emotion.

The way it works is by relying on a dictionary that maps lexical (aka word-based) features to emotion intensities,
these are known as sentiment scores. The overall sentiment score of a comment/post is achieved by summing up the
intensity of each word in the text. In some ways, it's easy: words like ‘love’, ‘enjoy’, ‘happy’, ‘like’ all convey a
positive sentiment. Also VADER is smart enough to understand the basic context of these words, such as “did not love”
as a negative statement. It also understands the emphasis of capitalization and punctuation, such as “ENJOY” which is
pretty cool. Phrases like “The acting was good , but the movie could have been better” have sentiments in both
polarities, which makes this kind of analysis tricky -- essentially w VADER you would analyze which part of the
sentiment here is more intense.

Possible improvements:
- A quant strategy has to have near perfect statistically significant results for it to be
relied on blinded, and this has a very narrow dataset that hasnt been tested across a full market cycle.
- Build a simple WSB focused wording database and reference each with abbreviations or numbers which will make it much easier
for code to understand
- Create a class that implements a common interface. This will allow you to swap out sentiment
analysers and also use different data sources by extending classes
- Implement it to know when to get it / out (execution)
- Use of an ML model (deep learning, transformers)
"""

import datetime as dt
import pandas as pd
import numpy as np
import praw
from nltk.sentiment.vader import SentimentIntensityAnalyzer as SIA
from keys import reddit_client_ID, reddit_secret_token, reddit_user_name, reddit_password
import re


def commentSentiment(ticker, urlT):
    subComments = []
    bodyComment = []
    try:
        check = reddit.submission(url=urlT)
        subComments = check.comments
    except:
        return 0

    for comment in subComments:
        try:
            bodyComment.append(comment.body)
        except:
            return 0

    sia = SIA()
    results = []
    for line in bodyComment:
        scores = sia.polarity_scores(line)
        scores['headline'] = line

        results.append(scores)

    df = pd.DataFrame.from_records(results)
    df.head()
    df['label'] = 0

    try:
        df.loc[df['compound'] > 0.1, 'label'] = 1
        df.loc[df['compound'] < -0.1, 'label'] = -1
    except:
        return 0

    averageScore = 0
    position = 0
    while position < len(df.label) - 1:
        averageScore = averageScore + df.label[position]
        position += 1
    averageScore = averageScore / len(df.label)

    return (averageScore)


def latestComment(ticker, urlT):
    subComments = []
    updateDates = []
    try:
        check = reddit.submission(url=urlT)
        subComments = check.comments
    except:
        return 0

    for comment in subComments:
        try:
            updateDates.append(comment.created_utc)
        except:
            return 0

    updateDates.sort()
    return (updateDates[-1])


def get_date(date):
    return dt.datetime.fromtimestamp(date)


# Is ticker a valid US company on 2021-06-02 ?
def is_valid_ticker(symbol):
    # https://quant.stackexchange.com/questions/1640/where-to-download-list-of-all-common-stocks-traded-on-nyse-nasdaq-and-amex#comment64621_1640
    stock_csv = pd.read_csv(r'C:\Users\thesi\OneDrive - CDPQ\Projet perso\stock_list.csv', header=None)
    stock_list = np.concatenate(stock_csv.values).tolist()
    is_us_ticker = symbol in stock_list # Is the symbol a US stock?

    return is_us_ticker


def get_top_mentioned(sub_reddit):
    # 25 most upvotes recently : log(abs(upvotes - downvotes)) + (now - timeposted /45000)
    # https://medium.com/hacking-and-gonzo/how-reddit-ranking-algorithms-work-ef111e33d0d9
    top_subreddit = sub_reddit.hot(limit=25)

    # Empty list of words
    words_collection = []

    # Collect words in submission titles
    for submission in top_subreddit:
        title = submission.title
        if title.isupper():  # Not all uppercase submission titles
            pass
        else:
            title_words = re.split("\s|,|;|:", title)
            words_collection.append(title_words)

    # Stock symbols in submission titles
    stock_symbols = []
    not_stocks = ["A", "I", "DD", "WSB", "YOLO", "RH", "EV", "PE", "ETH", "BTC", "E", "APES", "YOLO", "GAIN", "LOSS",
                  "PORN", "WILL", "NOT", "SELL", "AOC", "CNBC", "CEO", "IN", "DAYS", "DFV", "NEXT", "IT",
                  "SEND", "U", "MOON", "HOLD"]

    for title in words_collection:
        for word in title:
            # If word is upper case, does not contain a digit, length between 1 and 4 and is in US universe
            if word.isupper() and len(word) < 5 and not (
            any(char.isdigit() for char in word)) and word not in not_stocks:
                word = word.replace("$", "", 1)
                word = word.replace("#", "", 1)
                if is_valid_ticker(word):
                    stock_symbols.append(word)

    return list(set(stock_symbols))  # Remove duplicates when returning stock list


if __name__ == '__main__':
    # API keys
    reddit = praw.Reddit(client_id=reddit_client_ID,
                         client_secret=reddit_secret_token,
                         user_agent='SentAnalysis',
                         username=reddit_user_name,
                         reddit_password=reddit_password)

    # Top mentioned stocks
    sub_reddit = reddit.subreddit('wallstreetbets')
    stocks = get_top_mentioned(sub_reddit)
    print("Stock symbols: ", stocks)

    # submission_statistics = []
    # d = {}
    # for ticker in stocks:
    #     for submission in reddit.subreddit('wallstreetbets').search(ticker, limit=10):
    #         if submission.domain != "self.wallstreetbets":
    #             continue
    #         d = {}
    #         d['ticker'] = ticker
    #         d['num_comments'] = submission.num_comments
    #         d['comment_sentiment_average'] = commentSentiment(ticker, submission.url)
    #         if d['comment_sentiment_average'] == 0.000000:
    #             continue
    #         d['latest_comment_date'] = latestComment(ticker, submission.url)
    #         d['score'] = submission.score
    #         d['upvote_ratio'] = submission.upvote_ratio
    #         d['date'] = submission.created_utc
    #         d['domain'] = submission.domain
    #         d['num_crossposts'] = submission.num_crossposts
    #         d['author'] = submission.author
    #         submission_statistics.append(d)
    #
    # dfSentimentStocks = pd.DataFrame(submission_statistics)
    #
    # _timestampcreated = dfSentimentStocks["date"].apply(get_date)
    # dfSentimentStocks = dfSentimentStocks.assign(timestamp=_timestampcreated)
    #
    # _timestampcomment = dfSentimentStocks["latest_comment_date"].apply(get_date)
    # dfSentimentStocks = dfSentimentStocks.assign(commentdate=_timestampcomment)
    #
    # dfSentimentStocks.sort_values("latest_comment_date", axis=0, ascending=True, inplace=True, na_position='last')
    #
    # dfSentimentStocks.author.value_counts()
    #
    # dfSentimentStocks.to_csv('Reddit_Sentiment_Equity.csv', index=False)
