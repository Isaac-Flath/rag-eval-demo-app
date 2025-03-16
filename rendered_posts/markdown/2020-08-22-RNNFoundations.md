Isaac Flath

AboutBlogRSS

Subscribe

AboutBlogRSS

Subscribe

# NLP Recurrent NN Foundations

What is an RNN? How does NLP work?

## Contents

Credit Where Credit is DueData SetupGet the DataTokenizationNumericalizationSequence DefinitionDataloaderLanguage ModelNaive ModelRNN Number 1RNN Number 2Conclusion

[code]

    from fastai.text.all import *
[/code]

# Credit Where Credit is Due

The concept and techniques covered in this post are covered in much greater detail in Jeremy Howard and Sylvain Gugger's book. If you like this post, you should buy the book as you'll probably like it even more!

# Data Setup

### Get the Data

[code]

    path = untar_data(URLs.HUMAN_NUMBERS)
    lines = L()
    with open(path/"train.txt") as f: lines += L(*f.readlines())
    with open(path/"valid.txt") as f: lines += L(*f.readlines())
    lines
[/code]

(#9998) ['one \n','two \n','three \n','four \n','five \n','six \n','seven \n','eight \n','nine \n','ten \n'...]

### Tokenization

What is Tokenization?

Tokenization is about getting 'tokens' of language that have meaning. A word could be a token as it has meaning. A piece of punctuation could be a token as it has meaning. If a work is in all capital letters that could be a token. A portion of a word could be a token (ie dis) because a word beginning with dis has meaning. There are many many ways to tokenize, for this post I will use the most simple approach. That is, I will split based on spaces to make each word a token.

[code]

    txt = ' . '.join([l.strip() for l in lines])
[/code]

[code]

    tokens = L(*txt.split(' ')); tokens
[/code]

(#63095) ['one','.','two','.','three','.','four','.','five','.'...]

### Numericalization

Now that things are split into tokens, we need to start thinking about how to feed it to a Neural Network. Neural Networks rely on multiplication and addition, and we can't do that with a word. Somehow we need to convert these tokens to numbers. That is what Numericalization is all about. We will do this in a few steps:

  1. Get a unique list of all tokens (v)
  2. Assign a number to each of token (vocab)
  3. Replace tokens with numbers (nums)

[code]

    # Get a unique list of all tokens (v)
    v = tokens.unique()
    
    # Assign a number to each of token (vocab)
    vocab = {v:i for i,v in enumerate(v)};
    
    # We can lookup the number associated with a token like this
    vocab['fifty']
[/code]

23

[code]

    # Replace tokens with numbers (nums)
    nums = L(vocab[tok] for tok in tokens); nums
[/code]

(#63095) [0,1,2,1,3,1,4,1,5,1...]

### Sequence Definition

Now that we have tokens in the form of numbers, we need to create out inputs and outputs to the model. For this we need to organize our data into dependent and independent variables. Let's use the preceding 3 words to predict the next word. Below, we see the same thing in 2 ways - one with tokens and one with numbers. These are the same thing, just translating the tokens to numbers using the vocab above.

> #### ℹ️ Note
>
> Sequence Length (`sl`) will be 3, because we are using a sequence of 3 words to predict the next word.:
[code]

    sl = 3
    
    # For example, we will use the tokens 'one','.', and 'two' to predict '.'
    L((tokens[i:i+sl], tokens[i+sl]) for i in range(0,len(tokens)-sl-1,sl))
[/code]

(#21031) [((#3) ['one','.','two'], '.'),((#3) ['.','three','.'], 'four'),((#3) ['four','.','five'], '.'),((#3) ['.','six','.'], 'seven'),((#3) ['seven','.','eight'], '.'),((#3) ['.','nine','.'], 'ten'),((#3) ['ten','.','eleven'], '.'),((#3) ['.','twelve','.'], 'thirteen'),((#3) ['thirteen','.','fourteen'], '.'),((#3) ['.','fifteen','.'], 'sixteen')...]

[code]

    seqs = L((tensor(nums[i:i+sl]), nums[i+sl]) for i in range(0,len(nums)-sl-1,sl)); seqs
[/code]

(#21031) [(tensor([0, 1, 2]), 1),(tensor([1, 3, 1]), 4),(tensor([4, 1, 5]), 1),(tensor([1, 6, 1]), 7),(tensor([7, 1, 8]), 1),(tensor([1, 9, 1]), 10),(tensor([10, 1, 11]), 1),(tensor([ 1, 12, 1]), 13),(tensor([13, 1, 14]), 1),(tensor([ 1, 15, 1]), 16)...]

### Dataloader

Now we need to create our dataloader. The `dataloader` is just packaging it into batches, and not doing any transformations or changes to the data. What we saw above is what will be given to the model.

[code]

    bs = 128
    cut = int(len(seqs) * 0.8)
    dls = DataLoaders.from_dsets(seqs[:cut],seqs[cut:],bs=bs, shuffle=False)
[/code]

[code]

    dls2 = DataLoader(seqs[:cut],bs=bs, shuffle=False)
    dls3 = DataLoader(seqs[cut:],bs=bs, shuffle=False)
    
    dls4 = DataLoaders(dls3,dls3)
[/code]

# Language Model

### Naive Model

First, let's figure out a baseline for what is the best 'non-stupid' model we can come up with. If a model can't beat this score, then it's not worth anything.

The approach we will take will be to predict the most common token every time. If we do that we get about a 15% accuracy.

[code]

    n,counts = 0,torch.zeros(len(vocab))
    for x,y in dls.valid:
        n += y.shape[0]
        for i in range_of(vocab): counts[i] += (y==i).long().sum()
    idx = torch.argmax(counts)
    idx, v[idx.item()], counts[idx].item()/n
[/code]

(tensor(29), 'thousand', 0.15165200855716662)

### RNN Number 1

##### Code

We are going to make the simplest RNN we can. Here's a quick explanation of the code below.

`for i in range(sl):` Because we are feeding in a number of tokens based on our sequence length, sl, which was defined as 3. We will have 3 steps, 1 per token.

`h = h + self.i_h(x[:,i])` For each input token we will run our input to hidden function. We are indexing to grab the column in our embedding matrix that corresponds with the token, and adding that. All this is doing is adding the embedding for the particular token.

[code]

    
[/code]

`h = F.relu(self.h_h(h))` We then run our hidden to hidden function (h_h), which is a linear layer (y = wx + b). We do a ReLu of that, which is just replacing any negative values with 0.

[code]

    
[/code]

`return self.h_o(h)` We then run our hidden to output function (h_o), which is another linear layer, but it is outputing the prediction of which word is next. Naturally, this is the size of our vocabulary.

Wrap all that in a class and it looks like the below:

[code]

    class LM1(Module):
        def __init__(self, vocab_sz, n_hidden):
            self.i_h = nn.Embedding(vocab_sz, n_hidden)
            self.h_h = nn.Linear(n_hidden, n_hidden)
            self.h_o = nn.Linear(n_hidden,vocab_sz)
            
        def forward(self, x):
            h = 0
            for i in range(sl):
                h = h + self.i_h(x[:,i])
                h = F.relu(self.h_h(h))
            return self.h_o(h)
[/code]

Now we can run it below and see that we get almost 50% accuracy before we overfit, which is great considering the most common token only appears 15% of the time.

[code]

    learn = Learner(dls, LM1(len(vocab), 64), loss_func=F.cross_entropy, metrics=accuracy)
    learn.fit_one_cycle(4, 1e-3)
[/code]

epoch | train_loss | valid_loss | accuracy | time  
---|---|---|---|---  
0 | 2.505863 | 2.136583 | 0.458046 | 00:00  
1 | 1.602575 | 1.847033 | 0.480865 | 00:00  
2 | 1.503249 | 1.727588 | 0.492275 | 00:00  
3 | 1.436492 | 1.771485 | 0.410506 | 00:00  
  
##### Tensors

So what is it REALLY doing? To understand that, I find it helpful to think about the matrix/tensor sizes at each step.

**Embeddings**

Let's start with our input_hidden. Our Embedding matrix is has 64 weights (n_hidden) for each token in our vocabulary. So that looks like this:

$\underbrace{ \begin{bmatrix} \cdots & \cdots & \cdots & \cdots & \cdots & \cdots\  
\cdots & \cdots & \cdots & \cdots & \cdots & \cdots\  
\end{bmatrix}}_{\displaystyle 64-weights} \left.\vphantom{\begin{bmatrix} \cdots & \cdots & \cdots & \cdots & \cdots & \cdots\  
\cdots & \cdots & \cdots & \cdots & \cdots & \cdots\  
\end{bmatrix}}\right}30-tokens$

Now all the embedding layer does is get the correct columns. So for the first word in the sequence we get the index, then look it up in the embedding matrix. That 1 index location turns into the 64 weights.

$\underbrace{ \begin{bmatrix} \cdots \  
\cdots \ \cdots \  
\cdots \  
\cdots \  
\cdots \  
\end{bmatrix}}_{\displaystyle token-idx} \left.\vphantom{\begin{bmatrix} \cdots \  
\cdots \ \cdots \  
\cdots \  
\cdots \  
\cdots \  
\end{bmatrix}}\right}128-bs$ $==$ lookup in embedding matrix $==>$ $\underbrace{ \begin{bmatrix} \cdots & \cdots & \cdots & \cdots\  
\cdots & \cdots & \cdots & \cdots\  
\cdots & \cdots & \cdots & \cdots\  
\cdots & \cdots & \cdots & \cdots\  
\cdots & \cdots & \cdots & \cdots\  
\cdots & \cdots & \cdots & \cdots\  
\end{bmatrix}}_{\displaystyle 64} \left.\vphantom{\begin{bmatrix} \cdots & \cdots & \cdots & \cdots\  
\cdots & \cdots & \cdots & \cdots\  
\cdots & \cdots & \cdots & \cdots\  
\cdots & \cdots & \cdots & \cdots\  
\cdots & \cdots & \cdots & \cdots\  
\cdots & \cdots & \cdots & \cdots\  
\end{bmatrix}}\right}128$

**Hidden Linear Layer**

Next, we have out hidden_hidden. We have our 128x64 matrix from our embedding lookup and we need to do a linear layer.

$\underbrace{ \begin{bmatrix} \cdots & \cdots & \cdots & \cdots\  
\cdots & \cdots & \cdots & \cdots\  
\cdots & \cdots & \cdots & \cdots\  
\cdots & \cdots & \cdots & \cdots\  
\cdots & \cdots & \cdots & \cdots\  
\cdots & \cdots & \cdots & \cdots\  
\end{bmatrix}}_{\displaystyle 64-weights} \left.\vphantom{\begin{bmatrix} \cdots & \cdots & \cdots & \cdots\  
\cdots & \cdots & \cdots & \cdots\  
\cdots & \cdots & \cdots & \cdots\  
\cdots & \cdots & \cdots & \cdots\  
\cdots & \cdots & \cdots & \cdots\  
\cdots & \cdots & \cdots & \cdots\  
\end{bmatrix}}\right}128-bs$ $\underbrace{ \begin{bmatrix} \cdots & \cdots & \cdots & \cdots\  
\cdots & \cdots & \cdots & \cdots\  
\cdots & \cdots & \cdots & \cdots\  
\cdots & \cdots & \cdots & \cdots\  
\end{bmatrix}}_{\displaystyle 64} \left.\vphantom{\begin{bmatrix} \cdots & \cdots & \cdots & \cdots\  
\cdots & \cdots & \cdots & \cdots\  
\cdots & \cdots & \cdots & \cdots\  
\cdots & \cdots & \cdots & \cdots\  
\end{bmatrix}}\right}64$ $+$ $\underbrace{ \begin{bmatrix} \cdots & \cdots & \cdots & \cdots & \cdots & \cdots\  
\end{bmatrix}}_{\displaystyle 64-bias} \left.\vphantom{\begin{bmatrix} \cdots & \cdots & \cdots & \cdots & \cdots & \cdots\  
\end{bmatrix}}\right}1$ $=$ $\underbrace{ \begin{bmatrix} \cdots & \cdots & \cdots & \cdots\  
\cdots & \cdots & \cdots & \cdots\  
\cdots & \cdots & \cdots & \cdots\  
\cdots & \cdots & \cdots & \cdots\  
\cdots & \cdots & \cdots & \cdots\  
\cdots & \cdots & \cdots & \cdots\  
\end{bmatrix}}_{\displaystyle 64-weights} \left.\vphantom{\begin{bmatrix} \cdots & \cdots & \cdots & \cdots\  
\cdots & \cdots & \cdots & \cdots\  
\cdots & \cdots & \cdots & \cdots\  
\cdots & \cdots & \cdots & \cdots\  
\cdots & \cdots & \cdots & \cdots\  
\cdots & \cdots & \cdots & \cdots\  
\end{bmatrix}}\right}128-bs$ ===ReLu - Replace all negatives with 0 ===> $\underbrace{ \begin{bmatrix} \cdots & \cdots & \cdots & \cdots\  
\cdots & \cdots & \cdots & \cdots\  
\cdots & \cdots & \cdots & \cdots\  
\cdots & \cdots & \cdots & \cdots\  
\cdots & \cdots & \cdots & \cdots\  
\cdots & \cdots & \cdots & \cdots\  
\end{bmatrix}}_{\displaystyle 64-weights} \left.\vphantom{\begin{bmatrix} \cdots & \cdots & \cdots & \cdots\  
\cdots & \cdots & \cdots & \cdots\  
\cdots & \cdots & \cdots & \cdots\  
\cdots & \cdots & \cdots & \cdots\  
\cdots & \cdots & \cdots & \cdots\  
\cdots & \cdots & \cdots & \cdots\  
\end{bmatrix}}\right}128-bs$

And we do the above for however long our sequence is, in our case 3. So for each token we do the above. We start with 0 on the first loop, and each subsequent loop through we add onto that.

**Ouput Linear Layer**

We ended with a 128x64 matrix, which isn't exactly what we want. We have 30 words, so we want to know which one of the 30 is most likely. Specifically for each of the 128 items in our batch, we want 30 scores (1 for each word in our vocab). So we do a similar step as our hidden linear layer, but adjust the number of weights so we end up with the matrix of the appropriate size.

$\underbrace{ \begin{bmatrix} \cdots & \cdots & \cdots & \cdots\  
\cdots & \cdots & \cdots & \cdots\  
\cdots & \cdots & \cdots & \cdots\  
\cdots & \cdots & \cdots & \cdots\  
\cdots & \cdots & \cdots & \cdots\  
\cdots & \cdots & \cdots & \cdots\  
\end{bmatrix}}_{\displaystyle 64-weights} \left.\vphantom{\begin{bmatrix} \cdots & \cdots & \cdots & \cdots\  
\cdots & \cdots & \cdots & \cdots\  
\cdots & \cdots & \cdots & \cdots\  
\cdots & \cdots & \cdots & \cdots\  
\cdots & \cdots & \cdots & \cdots\  
\cdots & \cdots & \cdots & \cdots\  
\end{bmatrix}}\right}128-bs$ $\underbrace{ \begin{bmatrix} \cdots & \cdots\  
\cdots & \cdots\  
\cdots & \cdots\  
\cdots & \cdots\  
\end{bmatrix}}_{\displaystyle 30} \left.\vphantom{\begin{bmatrix} \cdots & \cdots\  
\cdots & \cdots\  
\cdots & \cdots\  
\cdots & \cdots\  
\end{bmatrix}}\right}64$ $+$ $\underbrace{ \begin{bmatrix} \cdots & \cdots & \cdots\  
\end{bmatrix}}_{\displaystyle 30-bias} \left.\vphantom{\begin{bmatrix} \cdots & \cdots & \cdots\  
\end{bmatrix}}\right}1$ $=$ $\underbrace{ \begin{bmatrix} \cdots & \cdots & \cdots & \cdots\  
\cdots & \cdots & \cdots & \cdots\  
\cdots & \cdots & \cdots & \cdots\  
\cdots & \cdots & \cdots & \cdots\  
\cdots & \cdots & \cdots & \cdots\  
\cdots & \cdots & \cdots & \cdots\  
\end{bmatrix}}_{\displaystyle 30-preds} \left.\vphantom{\begin{bmatrix} \cdots & \cdots & \cdots & \cdots\  
\cdots & \cdots & \cdots & \cdots\  
\cdots & \cdots & \cdots & \cdots\  
\cdots & \cdots & \cdots & \cdots\  
\cdots & \cdots & \cdots & \cdots\  
\cdots & \cdots & \cdots & \cdots\  
\end{bmatrix}}\right}128-bs$

### RNN Number 2

Now that we have a simple model, how do we improve it? There are many steps that need to be taken to get to a cutting edge model. We'll do one improvement, then leave the rest for future blog posts.

One thing that was a bit odd is in the training loop we reset back to 0 every time. What I mean by that, is we would loop through each of the 3 tokens, output our predictions for those, update the weights, then reset back for a new set. This isn't really how language works. Language has a pattern and a sequence to it. The further back you go the less important, but even things said a couple minutes ago could be important. Could you imagine holding a conversation if you could only remember and respond based on the last 3 words?

So let's fix this problem. We will move our h=0 up to the initialization of the class, and never reset back to 0. Instead, we will continuously keep adding to it. We will only update the last batch of weights (as if we updated all of them by the 1000th one we would be updating far to many weights to compute). We call this "detaching" it. Ultimately we are left with the same thing, but if has a memory of previous sequences beyond the one we are processing! Let's see if it makes things better.

[code]

    class LM2(Module):
        def __init__(self, vocab_sz, n_hidden):
            self.i_h = nn.Embedding(vocab_sz, n_hidden)
            self.h_h = nn.Linear(n_hidden, n_hidden)
            self.h_o = nn.Linear(n_hidden,vocab_sz)
            self.h = 0
    
        def forward(self, x):
            for i in range(3):
                self.h = self.h + self.i_h(x[:,i])
                self.h = F.relu(self.h_h(self.h))
            out = self.h_o(self.h)
            self.h = self.h.detach()
            return out
[/code]

To do this we need to take care that our data is in the appropriate order, so let's do a few tranformations to make that work.

[code]

    m = len(seqs)//bs
    m,bs,len(seqs)
    def group_chunks(ds, bs):
        m = len(ds) // bs
        new_ds = L()
        for i in range(m): new_ds += L(ds[i + m*j] for j in range(bs))
        return new_ds
[/code]

[code]

    cut = int(len(seqs) * 0.8)
    dls = DataLoaders.from_dsets(
        group_chunks(seqs[:cut], bs), 
        group_chunks(seqs[cut:], bs), 
        bs=bs, drop_last=True, shuffle=False)
[/code]

[code]

    learn = Learner(dls, LM2(len(vocab), 64), loss_func=F.cross_entropy, metrics=accuracy)
[/code]

[code]

    learn.fit_one_cycle(10, 3e-3)
[/code]

epoch | train_loss | valid_loss | accuracy | time  
---|---|---|---|---  
0 | 2.342321 | 1.897249 | 0.481689 | 00:00  
1 | 1.453624 | 1.713581 | 0.449707 | 00:00  
2 | 1.154838 | 1.680148 | 0.519775 | 00:00  
3 | 1.042766 | 1.566625 | 0.517822 | 00:00  
4 | 0.969852 | 1.633654 | 0.542480 | 00:00  
5 | 0.937066 | 1.581196 | 0.559570 | 00:00  
6 | 0.882712 | 1.660810 | 0.588379 | 00:00  
7 | 0.844926 | 1.595611 | 0.597656 | 00:00  
8 | 0.808309 | 1.613600 | 0.605225 | 00:00  
9 | 0.797358 | 1.621867 | 0.605713 | 00:00  
  
And we are up from about 50% accuracy to about 60%!

# Conclusion

Hopefully from this post you gained an understanding of the fundamental concepts behind NLP using Neural Networks. While this isn't cutting edge, the fundamental principles must be understood if you want to gain an intuition about what types of things might work.

### Stay Updated

Get notified about new posts on AI, web development, and tech insights.

Subscribe

## Contents

Credit Where Credit is DueData SetupGet the DataTokenizationNumericalizationSequence DefinitionDataloaderLanguage ModelNaive ModelRNN Number 1RNN Number 2Conclusion

## Let's Connect

GitHub

LinkedIn

Twitter

© 2025 Isaac Flath • All rights reserved

