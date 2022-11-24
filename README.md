# IMDB Review Sentiment Prediction

![image](https://user-images.githubusercontent.com/57106474/203849591-11a85ce4-77f0-4caa-b4d2-cec429f72fa2.png)

### Introduction
This is an implementation Logistic Regression with gradient descent implementation, used to perform sentiment analysis on IMDB reviews. Given a review, our task is to predict whether the person likes or dislikes the film, as well as determine the most important words in our analysis.

### Dataset
The dataset, provided by Stanford University, is a set of files that contain IMDB reviews of movies, accompanied by their rating scores appended to the filenames. There is also a vocabulary file provided which is a list of English words that are mentioned.

URL: http://ai.stanford.edu/~amaas/data/sentiment/aclImdb_v1.tar.gz

### Pre-processing

The data is vectorized using sklearn's Count Vectorizer, cutting the least used 1% and most frequently used 50%. Further, the most important features are selected based on their absolute co-variance, with a cut-off of 7. This choice was made on the basis that it yielded a balanced number of positive and negative features as well as reducing the number of features down to 347.

![image](https://user-images.githubusercontent.com/57106474/203850862-55a5b076-cd10-4d27-9399-f4de845ea5b7.png)


### The model

As mentioned earlier, this is a Logistic Regression model. In particular, the model uses cross-entropy cost as its loss function, as well as gradient descent for weight optimization, with a learning rate of .6 . It takes as input sparse matrices, which are better optimized for our purposes. The output of predictions is an array of labels within range [0;1], which can be rounded to be interpreted as exact prediction of sentiment where 1 means positive and 0 means negative.

The stopping conditions are maximum iterations or minimum norm of the gradient, which are set to 1e5 and 1e-5 respectively.


![image](https://user-images.githubusercontent.com/57106474/203850573-82ada414-9ce3-4f7a-880e-37c0c50df88f.png)



### Model Verification

Before experimenting we've done various verification experiments such as: verifying the gradient using small perturbation, visualizing cross-entropy loss as a function of time and examining the movement of weights as a function of time. These preliminaries have all shown that our model works as intended.

![image](https://user-images.githubusercontent.com/57106474/203851761-996330a4-e9e0-454e-b8d5-7f436c7e8005.png)


### Experiments and Results

Our experiments have resulted in 86% accuracy, which is a similar result compared to sklearn implementations of the same model, as well as kNN. However, our AUROC of .94 has greatly over-performed that of said implementations, demonstrating superior diagnostic ability.

![image](https://user-images.githubusercontent.com/57106474/203850248-de959f1a-71f6-40a2-b635-a51a8a072796.png)

Further experimentation has been done using different fractions of dataset, which showed a clear correlation in performance and amount of data utilized. In addition, experimentation with learning rates have shown that an increase in learning rate , within the range [0;1], improved time complexity with no effect on prediction performance.

![image](https://user-images.githubusercontent.com/57106474/203851524-6855bbf5-6be1-49dd-9043-c543345133af.png)


### Conclusion

All in all we've seen that using a logistic regression model with the right feature selection and data pre-processing methods yields great diagnostic ability for this kind of problem.

Further, we've seen that we can also predict exact ratings using simple linear regression, with a squared loss of less than one. This high accuracy however might stem from the bias of using only IMDB reviews.

Further exploration could be done in generalizing the model by training and testing on text that isn't strictly IMDB reviews, however the issue of availability of high-quality labelled data is an obstacle.
