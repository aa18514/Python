# ridge regression - netflix dataset
use verbosity (-v/-vv) to switch between naive linear regression and regression with L2 regularization <br> 
L2 regularization is used to reduce overfitting (https://en.wikipedia.org/wiki/Overfitting) and improve test accuracy <br> 
the dataset consits of 671 users and 9066 movies <br> 
all the data exists in the subdirectory "\movie-data" <br>
x, the feature vector is d + 1 dimension vector where d = 16 <br> 
y^, the target function is the expected rating <br> 
the loss function in this case is taken to be mean squared error between the predictor and the target function <br> 
training data(x[1:,]) is normalized to zero mean and unit variance, the same parameters are used to normalize the test data set <br>
x[0] = 1 <br> 
all of the expected ratings are quantized to the nearest 0.5 <br>
## Results
### cross validation error against K
![Alt text](https://github.com/aa18514/Python/blob/master/netflix_regression/cross_validation_error.png "Cross Validation Error versus K") <br>
### training & test error
![Alt text](https://github.com/aa18514/Python/blob/master/netflix_regression/test_train_error.png "training/test error for each User" ) <br>
mean test error: 1.868807 <br> 
mean train error: 0.531693 <br>
### training & test error
![Alt text](https://github.com/aa18514/Python/blob/master/netflix_regression/regression_without_regularization.png "training/test error for each User") <br> 
mean test error: 2.021839 <br>
mean train error: 0.514625 <br>
