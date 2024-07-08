# ML Design

## MACHINE LEARNING DESIGN INTERVIEW
_by Khang Pham_

## Table of contents

* [Chapter 1]
   * [Feature Selection and Feature Engineering](#feature-selection-and-feature-engineering)
     - [Categorical Features](#categorical-features)
    
## Chapter 1

## Feature Selection And Feature Engineering

### Categorical Features
* **The problem with One Hot Encoding**

Tree-based models, such as decision trees, random forests, and boosted trees, don’t perform well with one hot encodings, especially when the tree has many levels (i.e., when there are values of categorical attributes). This is because they pick the feature to split, based on how well splitting the data on that feature will “purify” it. If we have several levels, only a small fraction of the data will usually belong to any given level, so the one hot encoded columns will be mostly zeros. Since splitting on this column will only produce a small gain, tree-based algorithms typically ignore the information in favor of other columns. This problem persists, regardless of the volume of data you actually have. Linear models or deep learning models do not have this problem.

* **Best Practices**
When levels (categories) are not important, we can group them together in ”Other” class.

* Feature Hashing
Feature hashing, or hashing trick, converts text data, or categorical attributes with high cardinalities, into a feature vector of arbitrary dimensionality.
 <img src="featureHashing.png" width="300">

* Embedding
In practice, there are two ways to train embedding: pre-trained embedding i.e: word2vec2 style or cotrained, (i.e., YouTube video embedding).

There are two ways to formulate the problems: Continuous Bag of Words
(CBOW) and Skip-gram. For CBOW, we want to predict one word based on the surrounding words. For example, if we are given: word1 word2 word3 word4 word5, we want to use (word1, word2, word4, word5) to predict word3.
CBOW. Source: Exploiting Similarities Among Languages for Machine Translation
<img src="CBOW.png" width="500">

In the skip-gram model, we use ’word3’ to predict all surrounding words ’word1, word2, word4, word5’.
<img src="skip_gram.png" width="500">
