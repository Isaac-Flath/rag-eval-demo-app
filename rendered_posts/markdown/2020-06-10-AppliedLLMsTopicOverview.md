Isaac Flath

AboutBlogRSS

Subscribe

AboutBlogRSS

Subscribe

# Applied LLMs Topic Overview

A brief summary of the different topics that go into developing a good LLM based solution

## Contents

Play with your problemEvalsEval LevelsEval/Test SetsEval FrameworkSynthetic DataPrompt EngineeringRAGAgents/Self HealingFine TuningDeployment

In this post I will be discussing the applied side of LLMs and an overview of different techniques. You can think of this as a more detailed table of contents that will be the foundation of future posts. Each of these steps will be covered in future blog posts in more detail.

## Play with your problem

The first step is to play with your problem. This is where you start to build some intuition, and it's the one step you don't have to think super deeply about what you're doing.

Start with chatGPT or another chat interface and see how it does. Don't spend tons of time on prompt engineering, but follow things that make sense. See if things look promising and what kinds of things seem easy or hard for the model. Try to give it hard examples to break it. c

You don't have any evaluations yet, so all of this is very subjective and feeling based. Spend a couple hours on this step, but don't get too bogged down.

## Evals

Now that you've got some idea of what OOTB chat interface does, it's time to start building evaluations to be able to quantify how good a solution is (and by extension compare different solutions).

#### Vocab

**Eval** : A general term for something that gives info on how good a solution is at a particular task.

**Test** : An assertion like you may write in pytest or another testing framework. A test is a specific type of an eval.

**Metric:** A Metric is an aggregated meaasure of performance. It could be the percentage of times a test or eval passes, or a traditional metric such as precision or recall.

### Eval Levels

There are three levels of evals you should build. (See this article by Hamel Husain for more details):

  1. **Objective Test** : These are assertions, such as code to check if the output is valid JSON.
  2. **Subjective Eval** : These are opinions of a person or model, such as asking a model to say if a response is good or bad.
  3. **Production Eval** : These are business value oriented evals that cannot be calculated without deploying to production.

### Eval/Test Sets

It is critically important to have a test set that is separate from your training set. The test set needs to be as close to production data as the model will see, which means it cannot be data it was trained on.. This is normal best-practice in all of statistics and data science.

### Eval Framework

Evaluating the model to determine how it is behaving and whether it's improved is the most important thing. To do that effectively you need to look at the data, a lot. So you need this to be as frictionless as possible. There are many eval frameworks out there. If I were to pick one to use I would choose inspect_ai.

However, I recommend using Shiny for Python to build a custom web app to build your own custom eval UI. I believe evals to be the most important part of the process and it is always very specific to the use-case and so I believe the flexibility here is worth it. You cannot just use generic evals that are applied to every problem and be successfull. Shiny for python lets me create exactly what I need and bring in any relevant data, context, test results, etc. in the exact format I need. I highly recommend the core version and not the express version.

## Synthetic Data

To create your dataset for training or for testing, you can often create synthetic data with LLMs. This is helpful in several situations:

  * When you don't have any data and you want to prove out a concept quickly
  * When you have a little data and you want to augment it. For example changing out function names, facts, dates, in the prompt and response in a programatic way.
  * When you have a better model available. You can use llama3 70b to create training data to fine tune llama3 8b.

## Prompt Engineering

You need to get good at prompt engineering, because in some cases prompt engineering is sufficient. It can drastically change the quality of your outputs and can really make or break a project. Tech loves fancy sounding names, but they're pretty simple concepts. A few techniques:

  * **Chain of Thought (CoT):** Ask the model to think through and explan a problem before answering the question.
  * **Few Shot:** Add examples to your prompt. Especially ones the model gets wrong.
  * **Give reference material:** Manually add context to your prompt to help the model. Doing this in an automated way is called RAG.
  * **Write clearly:** Spend time explaining what you want it to do in an organized and concise way. Structure it with markdown section and bullets as needed. Give steps needed to solve the problem.

## RAG

Retrieval Augmented Generation (RAG) is about adding reference material to the prompt in an automated way. Reference material can be a lot of things, such as:

  * **Text** : Texts that are relevant to the ask, such as a paragraph or a page from a reference article.
  * **Code** : Code or functions that are relevant to the ask
  * **Data** : A table of data that contains the source data related to the question
  * **Schema** : Table schema's that are relevant or the user has access to.
  * **Examples** : The most similar K examples to the input to curate the best few-shot examples
  * **Metadata** : Metadata about file names, paths, modified people/times, etc. that is needed to answer a question

The idea is to fill in information that may not be in the original training data, but is needed to answer the question. This is often done in an automated way, but sometimes you need to do it manually.

#### > ⚠️ Important!

> It is important to understand what an embedding is to know what the limitations are for retrieving context. An embedding of a document will never help you answer what the last file user X modified or other time based questions

## Agents/Self Healing

All an agent means is that the LLM you prompt triggers another action (either calls a function or calls another model).

A simple agent could be produce an output, then call an eval on the output. If that eval fails, then reprompt the model to try to get a better response. This is a form of self-correction or self-healing.

## Fine Tuning

Once everything else is done you can think about fine-tuning. Fine tuning will add consistency and improve performance. But it generally offers a much smaller boost than RAG. Furthermore you should fine-tune AND use RAG, not just one or the other. So you need RAG in place to effectively fine-tune.

Fine-Tuning can be used for a few reasons:

  * **Most Common:** Improve perfomance on smaller model so you can use a cheaper faster model for your use-case
  * **Most Common:** Learn a domain where there is too much information to put in a prompt effectively (ie writing niche DSL code)
  * **Common:** Improve output format consistency.
  * **Rarely Needed:** Improve model perfomance by a few percent on your existing use case

There are many things to know about fine-tuning from Quantization, Lora, DPO, and more.

## Deployment

I believe that deployment is a software engineering function not a data science function. However, many data scientists want to deploy and share their own models (including me). This is challenging because it's a different discipline, but it is doable with the right tools.

For most data scientists that don't have full-stack development background here's what I recommend.

  * Proof of Concept Application: Python for Shiny
  * Model Deployment: Modal or Replicate
  * Web Application Development: fasthtml
  * Web application deployment: railway

If you learn both front-end and back-end development you will be able to be more creative applications. Most data scientists will get by just fine with the above, but if you want to be more versatile learning those other domains to some degree is very helpful. I recommend boot.dev to learn backend development. To learn front-end development, I don't have a specific course recommendation but you'll need to learn HTML, css, and javascript.

### Stay Updated

Get notified about new posts on AI, web development, and tech insights.

Subscribe

## Contents

Play with your problemEvalsEval LevelsEval/Test SetsEval FrameworkSynthetic DataPrompt EngineeringRAGAgents/Self HealingFine TuningDeployment

## Let's Connect

GitHub

LinkedIn

Twitter

© 2025 Isaac Flath • All rights reserved

