# ridge regression - netflix dataset

## Running the application 

use verbosity (-v/-vv/-vvv) to switch between naive linear regression, regression with L2 regularization and regression with transformed <br> 
features with regularization <br> 
## Dataset 

Dataset consits of 671 users and 9066 movies <br> 
All the data exists in the subdirectory "\movie-data" <br>

## Input features

The input features originally have a dimension of d + 1 where d = 18 and in this case x[0] = 1 <br> 
x[0] is used for the constant offset term, the rest of the terms represent different movie genres <br> 

## Output 

The predictor function, y_hat represents the expected rating <br>
The target function, y represents the actual rating <br> 

## Stratagies

### Prepocessing the input features

All feature vectors are normalized to zero and unit variance, the same parameters are used to normalize the test data set. <br>
The step is done before the data is partitioned according to different users, where for each user we derive an optimal weight vector <br>
In the case of non-transformed features the size of the feature vector is 19, and in the case of transformed features the size of <br>
the feature vector is 172 <br>

### Controlling Overfitting

L2 regularization is used to reduce overfitting (https://en.wikipedia.org/wiki/Overfitting) and improve test accuracy <br> 
We learn the regularized weights for each user seperately which leads to higher bias and lower bias as compared to taking a single <br> 
regularized weight vector for all the users <br> 
The following graph shows the lambda values for all 671 users: 
<p align="center"> 
	<img src="https://github.com/aa18514/Python/blob/master/netflix_regression/images/lambda_values.png" /> 
</p> 
the loss function in this case is taken to be L2 norm (Euclidean length between the predictor and the target) <br> 
Added support for the multi-processing module to parallelize against different values of K respectively <br> 
### Applying non-linear transformation
the non-linear transformation is taken to be each genre multiplied with the rest of the genres in the dataset. <br>
<p align="center"> 
	<img src="https://github.com/aa18514/Python/blob/master/netflix_regression/images/correlation_coefficients.png" />
</p>
The following figure shows the correlation coefficients between different genres in the dataset; it is worth noting <br> 
that the correlation coefficient between the genre 'Comedy' and 'Drama' is -0.61976, which shows that most of the movies <br>
that contain the genre 'Comedy' do not contain the genre 'Drama', and vice-versa  <br> 
the original features remain unchanged, the transformed features are appended to the original feature vector <br>
This yields a new dimension vector with a dimension of 172 <br>  
## Results
The following figures show the exponentially weighted training and test errors for 671 netflix users, which makes it more conveniant for us to capture the trends in the training and test bias. 
The exponentially weighted average is calculated as follows: 
<p align="center"> meanNext = beta * meanPrev + (1 - beta) * current_error </p> 
where meanPrev is initialized to zero <br> 
after each iteration the value if meanPrev is updated to the value of meanNext respectively <br>
the value of beta chosen for the analysis is 0.9, although in the future this can be experimented choosing an appropiate value of beta can be experimented with in the future <br>
taking the value of beta equal to 0.9 is analagous to taking the mean over the last 10 iterations <br>
<br>
<div>
	<img src="https://github.com/aa18514/Python/blob/master/netflix_regression/images/bias_against_K.png" width="400" height="400" />
	<img src="https://github.com/aa18514/Python/blob/master/netflix_regression/images/variance_against_K.png" width="400" height = "400" /> 
</div>

<p align="center"> 
	<img src="https://github.com/aa18514/Python/blob/master/netflix_regression/images/total_error.png" width="400" height="400" /> 
</p>
<div>
	<img src="https://github.com/aa18514/Python/blob/master/netflix_regression/images/test_train_error.png" width="400" height="400" />
	<img src="https://github.com/aa18514/Python/blob/master/netflix_regression/images/regression_without_regularization.png" width="400" height = "400" /> 
</div>
<br>
<p align="center"> 
	<img src="https://github.com/aa18514/Python/blob/master/netflix_regression/images/non-linear-features.png" width="400" height="400" /> 
</p>

|  | Mean Test Bias | Mean Train Bias | Mean Test Variance | Mean Train Variance |
| :---: | :-: | :-: | :-:| :-: |
| **Unregularized (Original Features)**  | 1.932715 | 0.564595 | 2.416220 | 0.130706 |
| **Regularized (Original Features)**    | 1.889901 | 0.572407 | 2.276397 | 0.129983 |
| **Regularized (Transformed Features)** | 1.300464 | 0.595549 | 0.770308 | 0.125738 | 