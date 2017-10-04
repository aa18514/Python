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
training data(x[1:,]) is normalized to zero mean and unit variance, the same parameters are used to normalize the test data set <br>
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
| **Unregularized (Original Features)**  | 1.567320 | 0.561700 | 1.809461 | 0.131361 |
| **Regularized (Original Features)**    | 1.510022 | 0.611829 | 1.417825 | 0.147195 |
| **Regularized (Transformed Features)** | 1.829337 | 0.277696 | 1.361984 | 0.061122 | 