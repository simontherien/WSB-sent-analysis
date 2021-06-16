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
- Empirical wsb_lexicon sentiment scores
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


# For debugging
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


# Sentiment score of submission
def get_submission_sentiment(url_t):
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
        scores = SIA.polarity_scores(line)  # Score for each comment
        scores['headline'] = line  # Comment
        results.append(scores)  # Append scores and comment to results

    df = pd.DataFrame.from_records(results)  # Results to pandas dataframe

    df['label'] = 0  # Label : positive, negative or neutral comment
    df.loc[df['compound'] > 0.05, 'label'] = 1  # Positive comment then label = 1
    df.loc[df['compound'] < -0.05, 'label'] = -1  # Negative comment then label = -1, else label = 0

    average_score = 0
    position = 0
    while position < len(df.label) - 1:
        average_score = average_score + df.label[position]  # Sum of all sentiment labels
        position += 1
    average_score = average_score / len(df.label)  # Average sentiment of comments in submission

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
        update_dates.append(comment.created_utc)  # Creation date of comment

    update_dates.sort()  # Increasing dates
    return update_dates[-1]  # Date of last comment


# Time stamp to date
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
    # 20 most upvotes recently : log(abs(upvotes - downvotes)) + (now - timeposted /45000)
    # https://medium.com/hacking-and-gonzo/how-reddit-ranking-algorithms-work-ef111e33d0d9
    top_subreddit = sub_reddit.hot(limit=20)

    # Empty list of words
    words_collection = []

    # Collect words in submission titles
    for submission in top_subreddit:
        title = submission.title
        title_words = re.split("\s|,|;|:|/", title)
        words_collection.append(title_words)

    # Stock symbols in submission titles
    stock_symbols = []
    not_stocks = ["A", "I", "DD", "WSB", "YOLO", "RH", "EV", "PE", "ETH", "BTC", "E", "APES", "YOLO", "GAIN", "LOSS",
                  "WILL", "NOT", "SELL", "AOC", "CNBC", "CEO", "IN", "DAYS", "DFV", "NEXT", "IT",
                  "SEND", "U", "MOON", "HOLD", "USD", "TD", "IRS", "ALL", "ON", "LOAN", "SI", "PSA", "ITM", "EM", "K",
                  "JP", "TA"]

    for title in words_collection:
        for word in title:
            # If word is upper case, does not contain a digit, length between 1 and 4 and is in US universe
            if word.isupper() and len(word) < 6 and not (
                    any(char.isdigit() for char in word)) and word not in not_stocks:
                word = word.replace("$", "", 1)
                word = word.replace("#", "", 1)
                if is_valid_ticker(word):
                    stock_symbols.append(word)

        stock_symbols_list = list(stock_symbols)
        stock_symbols_dict = dict((x, stock_symbols_list.count(x)) for x in set(stock_symbols_list))

    return dict(sorted(stock_symbols_dict.items(), key=lambda item: item[1], reverse=True))


# Weighted average of sentiment of submissions based on number of comments
def get_ticker_stats(sentiment_df):
    w_a = lambda x: np.average(x, weights=sentiment_df.loc[x.index, 'num_comments'])

    return sentiment_df.groupby('ticker').agg(ticker_ncomments=('num_comments', 'sum'),  # Total n of comments
                                              ticker_nupvotes=('score', 'sum'),  # Total score of ticker
                                              ticker_sentiment=(
                                              'comment_sentiment_average', w_a))  # W average of sentiment

def get_rank_ticker(stats_df):
    stats_df['comment_rank'] = stats_df['ticker_ncomments'].rank(ascending=False)
    stats_df['upvote_rank'] = stats_df['ticker_nupvotes'].rank(ascending=False)
    stats_df['sent_rank'] = stats_df['ticker_sentiment'].rank(ascending=False)

    # pd.factorize will generate unique values for each unique element of a iterable.
    # We only need to sort in the order we'd like, then factorize.
    # In order to do multiple columns, we convert the sorted result to tuples.
    cols = ['ticker_ncomments', 'ticker_nupvotes', 'ticker_sentiment']
    tups = stats_df[cols].sort_values(cols, ascending=False).apply(tuple, 1)
    f, i = pd.factorize(tups)
    factorized = pd.Series(f + 1, tups.index)
    stats_df['total_rank'] = factorized

    return stats_df


if __name__ == '__main__':
    # Top mentioned stocks
    sub_reddit = reddit.subreddit('wallstreetbets')
    stocks_dict = get_top_mentioned(sub_reddit)
    print("Stock symbols by occurrences in 20 hottest WSB posts : ", stocks_dict)

    df_wsb_words = pd.read_csv('wsb_lexicon.csv')
    wsb_words = list(df_wsb_words.itertuples(index=False, name=None))

    for word, score in wsb_words:
        add_wsb_word(word, score)

    stocks = list(stocks_dict.keys())

    submission_statistics = []  # list of dicts
    for ticker in stocks:
        # Search for top posts containing ticker in title and limit to 5
        for submission in reddit.subreddit('wallstreetbets').search(ticker, time_filter='month', limit=5):

            if submission.domain != "self.wallstreetbets":
                continue
            d = {}  # Initialize empty dict
            d['ticker'] = ticker  # Ticker column
            d['num_comments'] = submission.num_comments  # Number of comments in post
            d['comment_sentiment_average'] = get_submission_sentiment(
                submission.url)  # Mean of sentiment score of comments
            if d['comment_sentiment_average'] == 0.000000:  # Skip if submission sentiment is neutral
                continue
            d['score'] = submission.score
            d['creation date'] = get_date(submission.created_utc)  # Creation date of submission
            d['latest_comment_date'] = get_date(latest_comment(submission.url))  # Latest comment in submission
            d['author'] = submission.author  # Author of submission
            submission_statistics.append(d)

    dfSentimentStocks = pd.DataFrame(submission_statistics)
    dfSentimentStocks.to_csv('wsb_submission_analysis.csv', index=False)

    df_ticker_stats = get_ticker_stats(dfSentimentStocks).reset_index()
    df_ticker_stats_ranked = get_rank_ticker(df_ticker_stats)
    df_ticker_stats_ranked.to_csv('wsb_ranked_tickers.csv', index=False)
