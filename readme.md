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

#### Companies using word2vec method to train embedding?
- Instagram’s personalized recommendation model uses word2vec style where each user session can be viewed as: account 1 →\rightarrow account 2 →\rightarrow account 3 to predict accounts with which a person is likely to interact within a given session.
- Pinterest Ads ranking uses word2vec style where each user session can be viewed as: pin A →\rightarrow pin B →\rightarrow pin C, then co- trained with multitask modeling.
- DoorDash personalized store feed uses word2vec style where each user session can be viewed as: restaurant 1 →\rightarrow restaurant 2 → \rightarrow restaurant 3. This Store2Vec model can be trained to predict if restaurants were visited in the same session using CBOW algorithm.

#### How does DoorDash Train Embedding?
For each session, we assume users may have a certain type of food in mind, and they view store A, store B, etc. We can assume these stores are somewhat similar to the user’s interests. We can train a model to classify a given pair of stores if they show up in a user session. 
<img src="embedding_DoorDash.png" width="700">


#### How does YouTube Train Embedding?

Recommendation System usually consists of three stages: **Retrieval, Ranking and Re-ranking** (read Chapter [rec-sys]). In this example, we will cover how YouTube builds **Retrieval** (Candidate Generation) component using Two- tower architecture.

We have two towers: left tower takes (users, context) as input and right tower takes movies as input.
- Given input x (user, context), we want to pick candidate y (videos) from all available videos.
- A common choice is to use Softmax function
$$P(y| x; \theta) = \frac{e^{s(x, y)}}{\sum_{i=1} e^{s(x, y_i)}}$$
- Loss function: use log-likelihood $$L = - \frac{1}{T} \sum_{i=1}^T \log(P(y_i|x_i;\theta))$$
- As a result, the two-tower model architecture is capable of modeling the situation where the label has structures or content features.
- StringLookup api maps string features to integer indices.
- Embedding layer API turns positive integers (indexes) into dense vectors of fixed size.
<img src="embedding_YouTube.png" width="700">

* Key Questions:
  - inventory too huge => Solution: for each mini-batch, we sampled data from our videos corpus as negative samples. One example is to use power-law distribution for sampling.
  - When sampling, it’s possible that popular videos are overly penalized as negative samples in a batch. Does it introduce bias in our training data? One solution is to “correct” the logit output $$sc(xi,yj)=s(xi,yj) −\log(p_j)s^c(x_i, y_j) = s(x_i, y_j) - \log(p_j)$$. Here $p_j$ means the probability of selecting video j. 
    
#### How does LinkedIn Train Embedding?

LinkedIn used Hadamard product for Member Embedding and Job Embedding.
The final prediction is a logistic regression on the Hadamard product between each seeker and job posting pair.
<img src="embedding_LinkedIn.png" width="700">


$$[1234]⊙[5326]=[56624]$$

$$  
\begin{equation}
  \begin{aligned}
    \begin{bmatrix}
    1 & 2\\ 
    3 & 4 
    \end{bmatrix}
    \odot 
    \begin{bmatrix} 
    5 & 3 \\
    2 & 6 
    \end{bmatrix} = 
    \begin{bmatrix} 
    5 & 6\\ 
    6 & 24 
    \end{bmatrix}
  \end{aligned}
\end{equation}
$$

#### How does Pinterest Train Embedding?
When users search for a specific image, Pinterest uses input pins visual embedding and search for similar pins. How do we generate visual embedding? Pinterest used image recognition deep learning architecture, e.g., VGG16, ResNet152, Google Net, etc., to fine tune on the Pinterest dataset. The learned features will then be used as embedding for Pins. 

### How do we Evaulate the Quality of embedding?

There are two methods:
* Apply embedding to downstream tasks and measure their model performance. For certain applications, like natural language processing (NLP), we can also visualize embeddings using t-SNE (t-distributed stochastic neighbor embedding), EMAP.
* Apply clustering (kmeans, k-Nearest Neighbor) on embedding data and see if it forms meaningful clusters.

### Measuring Similarities

- Dot Product
- Cosine
- Euclidian
  
* Dot product tends to favor embeddings with high norm. It’s more sensitive to the embeddings norm compared to other methods. Because of that it can create some consequences
Popular content tends to have higher norms, hence ends up dominating the recommendations. 
