# https://github.com/cjhutto/vaderSentiment#python-demo-and-code-examples
# https://medium.com/ro-data-team-blog/nlp-how-does-nltk-vader-calculate-sentiment-6c32d0f5046b#id_token=eyJhbGciOiJSUzI1NiIsImtpZCI6IjE3MTllYjk1N2Y2OTU2YjU4MThjMTk2OGZmMTZkZmY3NzRlNzA4ZGUiLCJ0eXAiOiJKV1QifQ.eyJpc3MiOiJodHRwczovL2FjY291bnRzLmdvb2dsZS5jb20iLCJuYmYiOjE2MjI0Njc0MzIsImF1ZCI6IjIxNjI5NjAzNTgzNC1rMWs2cWUwNjBzMnRwMmEyamFtNGxqZGNtczAwc3R0Zy5hcHBzLmdvb2dsZXVzZXJjb250ZW50LmNvbSIsInN1YiI6IjExNjE0NTU3MDMyODg2ODcxMzUxMiIsImVtYWlsIjoic2ltb24udGhlcmllbkBnbWFpbC5jb20iLCJlbWFpbF92ZXJpZmllZCI6dHJ1ZSwiYXpwIjoiMjE2Mjk2MDM1ODM0LWsxazZxZTA2MHMydHAyYTJqYW00bGpkY21zMDBzdHRnLmFwcHMuZ29vZ2xldXNlcmNvbnRlbnQuY29tIiwibmFtZSI6IlNpbW9uIFRoZXJpZW4iLCJwaWN0dXJlIjoiaHR0cHM6Ly9saDMuZ29vZ2xldXNlcmNvbnRlbnQuY29tL2EtL0FPaDE0R2dNNUJ1MTVnZWMyZ2pIZGlNQm9XQlA2SUVJZlE2YkE3emFJSmQ1QWc9czk2LWMiLCJnaXZlbl9uYW1lIjoiU2ltb24iLCJmYW1pbHlfbmFtZSI6IlRoZXJpZW4iLCJpYXQiOjE2MjI0Njc3MzIsImV4cCI6MTYyMjQ3MTMzMiwianRpIjoiYzM2M2UwY2I3NTFkZTM1MjY2NTg3Yzc2MTlhNDc1MTMzMTdlNDA4OSJ9.RXROeWiqhiVxLO_Y14wBxEVq7tF_VJj22paCrMXEqjXFRgsVr5iDfZvPLEgyzKKrVjVLWddsNmupsI212enWU3Lz7UCds2DksfCvXzzNWgsAaN3lC1Hh1QxBjGEQCMx2-ci0G5TYnzjfU6tiZXPGeg8cUdkhUbEB7hf7ClcuBmdTpKHpkxz-hnBEcnSf_Af7Q9rYbgdBEc3Q6WUZSLiQ6jcrfW_qN2hCWHaCiTSicC9RwK7XlM_Jj__9bpMOEoqguu4jF_C7wFO0Jju1sfXMRXcrlHpQxui3uh5sEkDcSGgnlxDG4XcTgy7ev75TCgkIYCS8c5kszhOJaq-b-o1rqg

"""
Momentum based strategy

Scraped WSB sentiment, got the top + most positively mentioned stocks on WSB. Uses VADER (Valence Aware Dictionary
for Sentiment Reasoning), which is a model used for text sentiment analysis that is sensitive to both polarity
(positive/negative) and intensity (strength) of emotion.

Possible improvements:
- A quant strategy has to have near perfect statistically significant results for it to be
relied on blinded, and this has a very narrow dataset that hasn't been tested across a full market cycle.
- Create a class that implements a common interface. This will allow you to swap out sentiment
analysers and also use different data sources by extending classes
- Implement it to know when to get it / out (execution)
- wsb_lexicon sentiment scores based on historical returns (multivariate regression analysis : returns = B*words + u)
"""

import datetime as dt
import pandas as pd
import numpy as np
import praw
from praw.models import MoreComments
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from keys import reddit_client_ID, reddit_secret_token, reddit_user_name, reddit_password
import re

# Global variables
SIA = SentimentIntensityAnalyzer()
reddit = praw.Reddit(client_id=reddit_client_ID,  # API Keys
                     client_secret=reddit_secret_token,
                     user_agent='SentAnalysis',
                     username=reddit_user_name,
                     reddit_password=reddit_password)


# WSB focused wording database
def add_wsb_word(word, score):
    # Add word to lexicon with score between -4 and 4
    SIA.lexicon.update({word: score})


def get_sentence_classification(sentence):
    tokenized_sentence = nltk.word_tokenize(sentence)

    pos_word_list = []
    neu_word_list = []
    neg_word_list = []

    for word in tokenized_sentence:
        if (SIA.polarity_scores(word)['compound']) >= 0.1:
            pos_word_list.append(word)
        elif (SIA.polarity_scores(word)['compound']) <= -0.1:
            neg_word_list.append(word)
        else:
            neu_word_list.append(word)

    return dict([('Positive', pos_word_list), ('Neutral', neu_word_list), ('Negative', neg_word_list)])


def comment_sentiment(url_t):
    # Initialize empty list of body comments
    body_comment = []

    # Interact with CommentForest (list of top-level comments each of which contains a CommentForest of replies) of
    # submission object
    sub_comments = reddit.submission(url=url_t).comments

    # Output only the body of the top level comments in the thread
    for comment in sub_comments:
        # If comment forest contains a number of More Comments objects
        if isinstance(comment, MoreComments):
            continue
        body_comment.append(comment.body)

    results = []
    for line in body_comment:
        # Return a float for sentiment strength based on the input text. Positive values are positive valence,
        # negative value are negative valence
        scores = SIA.polarity_scores(line)
        scores['headline'] = line
        results.append(scores)
        print(scores)

    df = pd.DataFrame.from_records(results)
    df.head()
    df['label'] = 0

    df.loc[df['compound'] > 0.1, 'label'] = 1
    df.loc[df['compound'] < -0.1, 'label'] = -1

    average_score = 0
    position = 0
    while position < len(df.label) - 1:
        average_score = average_score + df.label[position]
        position += 1
    average_score = average_score / len(df.label)

    return average_score


def latest_comment(url_t):
    # Initialize empty list of dates
    update_dates = []

    # Interact with CommentForest (list of top-level comments each of which contains a CommentForest of replies) of
    # submission object
    sub_comments = reddit.submission(url=url_t).comments

    # Output only the body of the top level comments in the thread
    for comment in sub_comments:
        # If comment forest contains a number of More Comments objects
        if isinstance(comment, MoreComments):
            continue
        update_dates.append(comment.created_utc)

    update_dates.sort()
    return update_dates[-1]


def get_date(date):
    return dt.datetime.fromtimestamp(date)


# Is ticker a valid US company on 2021-06-02 ?
def is_valid_ticker(symbol):
    # https://quant.stackexchange.com/questions/1640/where-to-download-list-of-all-common-stocks-traded-on-nyse-nasdaq-and-amex#comment64621_1640
    stock_csv = pd.read_csv(r'us_stock_list.csv', header=None)
    stock_list = np.concatenate(stock_csv.values).tolist()
    is_us_ticker = symbol in stock_list  # Is the symbol a US stock?

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
                  "WILL", "NOT", "SELL", "AOC", "CNBC", "CEO", "IN", "DAYS", "DFV", "NEXT", "IT",
                  "SEND", "U", "MOON", "HOLD", "USD", "TD", "IRS", "ALL", "ON", "LOAN", "SI"]

    for title in words_collection:
        for word in title:
            # If word is upper case, does not contain a digit, length between 1 and 4 and is in US universe
            if word.isupper() and len(word) < 5 and not (
                    any(char.isdigit() for char in word)) and word not in not_stocks:
                word = word.replace("$", "", 1)
                word = word.replace("#", "", 1)
                if is_valid_ticker(word):
                    stock_symbols.append(word)

        stock_symbols_list = list(stock_symbols)
        stock_symbols_dict = dict((x, stock_symbols_list.count(x)) for x in set(stock_symbols_list))

    return dict(sorted(stock_symbols_dict.items(), key=lambda item: item[1], reverse=True))


if __name__ == '__main__':
    # Top mentioned stocks
    sub_reddit = reddit.subreddit('wallstreetbets')
    stocks_dict = get_top_mentioned(sub_reddit)
    print("Stock symbols by occurrences in 25 hottest WSB posts : ", stocks_dict)

    df_wsb_words = pd.read_csv('wsb_lexicon.csv')
    wsb_words = list(df_wsb_words.itertuples(index=False, name=None))

    for word, score in wsb_words:
        add_wsb_word(word, score)
    # #
    # # stocks = list(stocks_dict.keys())
    # #
    submission_statistics = []
    d = {}
    # for ticker in stocks:
    # Search for top posts containing ticker in title and limit to 5
    for submission in reddit.subreddit('wallstreetbets').search('BB', sort='top', time_filter='month', limit=5):

        if submission.domain != "self.wallstreetbets":
            continue
        d = {}  # Initialize empty dict
        d['ticker'] = 'BB'  # Ticker column
        d['num_comments'] = submission.num_comments  # Number of comments in post
        d['comment_sentiment_average'] = comment_sentiment(submission.url)  # Mean of sentiment score of comments
        if d['comment_sentiment_average'] == 0.000000:
            continue
        d['latest_comment_date'] = latest_comment(submission.url)
        d['score'] = submission.score
        d['upvote_ratio'] = submission.upvote_ratio
        d['date'] = submission.created_utc
        d['num_crossposts'] = submission.num_crossposts
        d['author'] = submission.author
        submission_statistics.append(d)

    dfSentimentStocks = pd.DataFrame(submission_statistics)

    _timestampcreated = dfSentimentStocks["date"].apply(get_date)
    dfSentimentStocks = dfSentimentStocks.assign(timestamp=_timestampcreated)

    _timestampcomment = dfSentimentStocks["latest_comment_date"].apply(get_date)
    dfSentimentStocks = dfSentimentStocks.assign(commentdate=_timestampcomment)

    dfSentimentStocks.sort_values("latest_comment_date", axis=0, ascending=True, inplace=True, na_position='last')

    dfSentimentStocks.author.value_counts()

    dfSentimentStocks.to_csv('WSB_Sent_Analysis.csv', index=False)
    #print(dfSentimentStocks)
