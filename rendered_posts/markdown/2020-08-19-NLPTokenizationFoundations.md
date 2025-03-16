Isaac Flath

AboutBlogRSS

Subscribe

AboutBlogRSS

Subscribe

# NLP Tokenization Foundations

What is Tokenization? What is numericalization?

## Contents

IntroCredit Where Credit is DueThe DataTokenizationA simple ApproachNext Steps

# Intro

In this post we are going to dive into NLP, specifically Tokenization. Tokenization are the foundation of all NLP.

So what is a language model? In short, it is a model that uses the preceding words to predict the next word. We do not need separate labels, because they are in the text. This is training the model on the nuances of the language you will be working on. If you want to know if a tweet is toxic or not, you will need to be able to read and understand the tweet in order to do that. The language model helps with understanding the tweet - then you can use that model with those weights to tune it for the final task (determining whether the tweet is toxic or not).

For this post, I will be using news articles to show how to tokenize a news article and numericalize it to get ready for deep learning.

# Credit Where Credit is Due

The concept and techniques covered in this post are covered in much greater detail in Jeremy Howard and Sylvain Gugger's book. If you like this post, you should buy the book as you'll probably like it even more!

# The Data

I will be using the "All-the-news" dataset from this site. https://components.one/datasets/all-the-news-2-news-articles-dataset/

I downloaded then put the csv into a sqlite database for convenience

[code]

    import pandas as pd
    import sqlite3
    con = sqlite3.connect('../../../data/news/all-the-news.db')
    
    
    pd.read_sql_query('SELECT publication, min(date),max(date), count(*) from "all-the-news-2-1" group by publication order by max(date) desc limit 5', con)
[/code]

| publication | min(date) | max(date) | count(*)  
---|---|---|---|---  
0 | Buzzfeed News | 2016-02-19 00:00:00 | 2020-04-02 00:00:00 | 32819  
1 | The New York Times | 2016-01-01 00:00:00 | 2020-04-01 13:42:08 | 252259  
2 | Business Insider | 2016-01-01 03:08:00 | 2020-04-01 01:48:46 | 57953  
3 | Washington Post | 2016-06-10 00:00:00 | 2020-04-01 00:00:00 | 40882  
4 | TMZ | 2016-01-01 00:00:00 | 2020-04-01 00:00:00 | 49595  
  
I am going to pick the 5 most recent New York times Articles. For the final model I will use all of the data, but for simplicity of demonstrating tokenization we will use just 5 articles. Here is an example of the start of one of the articles

[code]

    df = pd.read_sql_query('SELECT article from "all-the-news-2-1" where publication = "The New York Times" order by date desc limit 5', con)
    ex = df.iloc[1,0]; ex[:162]
[/code]

President Trump told of “hard days that lie ahead” as his top scientific advisers released models predicting that the U.S. death toll would be 100,000 to 240,000.

# Tokenization

So how do I turn what I see above (text) into something a neural network can use? The first layer in a neural network is going to do matrix multiplication and addition. How do I multiply "President Trump told of “hard days that lie ahead” as his top scientific advisers released models" by any number? This is the core question we will answer with tokenization.

> #### 💡 Tip
>
> Tokenization is the method in which we take text and turn them into numbers we can feed into a model

### A simple Approach

Let's start with a simple idea. Let's treat each word as separate inputs in the same way that separate pixels in an image are separate inputs. We can do this in the english language by splitting our text by spaces/

[code]

    ex[:162]
[/code]

President Trump told of “hard days that lie ahead” as his top scientific advisers released models predicting that the U.S. death toll would be 100,000 to 240,000.

[code]

    import numpy as np
    tokens = ex.split(sep = ' ')
    tokens[:10]
[/code]

['President', 'Trump', 'told', 'of', '“hard', 'days', 'that', 'lie', 'ahead”', 'as']

That's better, now we have distinct data points. But we need them to be numbers in order to multiply and add them. So let's replace each work with a number.

To do this we will get a unique list of all of the words, then assign a number to each word.

[code]

    from fastai2.text.all import *
    vocab = L(tokens).unique()
[/code]

[code]

    word2idx = {w:i for i,w in enumerate(vocab)}
[/code]

We have 20165 words, but only 1545 unique words. Each of those assigned a number in a dictionary.

[code]

    len(ex),len(vocab)
[/code]

(21065, 1545)

We can see that each word gets a number.

[code]

    list(word2idx.items())[:5]
[/code]

[('President', 0), ('Trump', 1), ('told', 2), ('of', 3), ('“hard', 4)]

Now all we have to do is replace our tokens with the numbers in our word2idx dictionary. Lets take a look at 10 words near the end of our article and see what it looks like as tokens as well as numbers

[code]

    nums = L(word2idx[i] for i in tokens)
    nums[3000:3010],L(tokens[3000:3010])
[/code]

((#10) [1359,24,17,943,1360,1361,388,331,77,1362], (#10) ['linked','to','the','coronavirus.','Only','Italy','has','recorded','a','worse'])

### Next Steps

While this is the idea behind tokenization, there are many things that were not considered. Here are some other ideas to consider when choosing a tokenization approach.

  * What holds meaning other than words in english that we could make into tokens? What about punctuation or a comma? What about the beginning of a sentence or paragraph?
  * Should 'words' and 'word' be tokenized as 2 separate words? Or could we assign 'word' and 's' as the tokens because the base of the word has the same meaning?
  * Is there a better way to break up a sentence other than by words? What if it were just based on common sentence strings. Maybe 'of a' could be 1 token rather than 2. could ' dis' or 'ing' be tokens that can be combined with many different words?

### Stay Updated

Get notified about new posts on AI, web development, and tech insights.

Subscribe

## Contents

IntroCredit Where Credit is DueThe DataTokenizationA simple ApproachNext Steps

## Let's Connect

GitHub

LinkedIn

Twitter

© 2025 Isaac Flath • All rights reserved

