![Header Graphic]()
# Analyzing Information Propagation using NLP and Machine Learning
Author: Blake McMeekin

## Overview

Information propagation is one of the fundamental forces of nature shaping our world. Whether you hear about a new idea or company, whether your customers hear about you, or whether your research paper gets cited, all ultimately depend on the way information is communicated and shared. Political campaigns, misinformation, company cultures and social revolutions all have one thing in common: some idea or set of information has propagated. All of culture, business, and technology can be seen and possibly measured through this lens.

If, on watching a new TV show, you share it with more than one other person, and they each share it with more than one person, and that pattern continues, that show will spread exponentially. This propagation rate (spreading more than once on average) is referred to as R0 in epidemiology. ["The R0 value of COVID-19 as initially estimated by the World Health Organization (WHO) was between 1.4 and 2.4"](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC7751056/) Anything that propagates (organisms, viruses, information, research) has an R0 value.

In this project, we see if we can use Natural Language Processing and Machine Learning to predict viral text in machine learning communities on twitter.

For more information about this topic generally, I recommend reading this [excellent post]() by Kevin Simler, or listening to [this podcast]() by ####.

## Organizational Problems

Business case:
Every organization depends on information propagation in some way. Often this is done through marketing expenditures which spread awareness of a product or service. When a product or service has a good R0 value, we say it has good "word of mouth" which lowers the customer acquisition cost and increases demand. If we could predict which things will go viral, this could economically generate demand for products or services.

Academic case:
Whether your research gets attention, citations, and funding ultimately depends on whether people hear about and share your work. For this reason every academic spends a good deal of time thinking about the effectiveness of their scientific communications, because their research impact will be a function of R0. If we can better predict which things will spread through an academic community, we might be able to steer scientific communications and research to reflect the interests of that community.

Covid-19 could be considered a case-study in what can happen when academic communications fail to rapidly translate to effective policy in the broader cultural context. Sharing research with the general public is of great importance for policymaking, funding, and public support, as well as for attracting new talent.

## Data

Data collection was done by looking at all of the followers/follows of machine learning researchers/educators (specifically Juergen Schmidhuber & Andrew Ng) and scraping tweets from those accounts. This could be taken further by looking at the intersection of followers of a few "celebrity" names within a particular niche, but this technique was not utilized here and we could therefore think of this analysis as slightly less targeted.

## Methods

Our analysis consisted of two main stages:
- NLP Pipelines for converting text to features for Machine Learning
- Machine learning models on NLP and tweet metadata

For our text data, three NLP processes were used: Bag of Words, Document Embeddings (via ROBERTA-Large), and Top2Vec. 

1.) - Bag of Words is the simplest here, counting word frequencies in documents. Before this, the text is cleaned and simplified.

2.) - Document Embeddings leverages the principle components of a large language model - in this case, ROBERTA-Large. These principle components reflect the 1024 most-descriptive dimensions in a larger corpus of text. We can convert our text data into these dimensions and use it to differentiate text styles.

3.) - Top2Vec is another layer on top of document embeddings - Top2Vec is the combination of a document embedding pipeline with a clustering algorithm, which identifies distinct clusters of documents and assigns them a label. Top2Vec is very interpretable, we can see what these topics are and which words tend to occur within them. two  examples in our analysis were OpenAI and posts hiring PHD candidates.

After our NLP pipeline, we have about 2500 features which can be reduced based on correlation with our target or SequentialFeatureSelection. For our regression we took the top 256 features based on correlation, while for classification we just went ahead and fed the full 2500 features into our machine learning.

Ultimately 6 machine learning algorithms were tried:
- Linear and Logistic Regression
- Random Forest
- XGBoost
- Deep Learning (cut from notebooks)
- 1D CNN (cut from notebooks)
- TabNet

Deep Learning and 1D CNNs were eventually cut, as they consistently underperformed compared to TabNet; TabNet is a deep learning algorithm developed by Google specifically for working with tabular data. 1D CNNs are an interesting approach however, converting text data into a simplified form of "image" in order to recognize patterns in data. In some cases this can perform well, but this did not appear to be one of those cases.

## Validation

For validating our models, we split our data into train, validation and test sets. For classification, the sizes of these sets were 23310, 6475, and 2590 respectively. Training was performed using training and validation sets, validation set was used to compare models, and then after a final model was chosen, our test set was used to confirm our results. Our Random Forest model went from an F1 of .31 to an F1 of .26, which is not bad considering we only have 70 positive targets in our test set.

## Findings

Viral text is definitely different in some ways from normal text. Some topics are clearly more viral, e.g. talking about OpenAI or hiring PHD candidates. With our approach, we could specifically quantify the virality of different words or topics - for example, the phrase "100daysofcode" had a 16% correlation with retweets in our machine learning niche.

Our classification was 97% accurate at classifying tweets as over or under 50 retweets, having an F1 score of 0.26. We were able to use NLP and machine learning to explain about 11% of what makes people share text (R2).

Our best performing algorithm for regression was XGBoost, while our best performing algorithm for classification was a Random Forest. I had high hopes for TabNet in interpreting our 2500 features but this failed to beat XGBoost on our traditional feature set.

ROC Curve for classification:
![ROC Curve]()

## Conclusion

Virality may always be hard to predict – like predicting the stock market, if you find something that works, it changes the landscape and probably stops working. We were able to predict some

## Next Steps

- Integrating our model with business processes or marketing dashboards may be the most obvious use-case
- Building a dashboard for people to play with could be a way to draw attention to this project
- Better NLP and document embeddings from newer large language models
- Reuse this NLP pipeline to make other predictions from text

## For More Information

Check out the code in the [regression]() or [classification]() notebooks, or review the [presentation]().

For additional information, contact Blake McMeekin at blakemcme@gmail.com

## Repository Structure
'''
├── data
│   └── combined_data.csv
├── graphics
├── Data_Sourcing_and_Processing_from_Twitter.ipynb
├── Roberta_data_processing.ipynb
├── Top2Vec_Data_Processing.ipynb
├── Viral_Classification.ipynb
├── Information Propagation Analysis.ipynb
├── Viral Text Prediction Presentation.pdf
└── README.md
'''
