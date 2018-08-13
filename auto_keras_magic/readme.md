# Machine Learning Using AutoKeras

## Installation
If installed from source, a dependency of AutoKeras is to install PyTorch==0.4.0, which means for those of you who are using PyTorch==0.4.1, the command 'pip install autokeras' may not work. You can get around this by changing the dependency in setup.py (line 6) from 'torch==0.4.0' to 'torch==0.4.1'. <br>
Alternativately you can use the 'autokeras.tar.gz' file in this repository to install the package. Simply run 'pip install autokeras.tar.gz' to install autokeras.
## Description
Use of an open source library [AutoKeras](https://autokeras.com/) for automated machine learning.
AutoKeras provides the functions to automatically search for architecture
and parameters for deep learning models. [Olivetti Faces Dataset](http://scikit-learn.org/stable/datasets/olivetti_faces.html)
was used for evaluation of the library. <br>
Because data was sampled from 40 individuals and each class only had 10 samples, the size of the dataset was increased by 4x by artificially synthesizing data, this was achived by rotating each image 90, 180 and 270 degrees counter clockwise <br>
The train test split was kept at 80:20. <br>

## Results

<br>
|  | Mean Test Bias | Mean Train Bias | Mean Test Variance | Mean Train Variance | time (s) |
| :---: | :-: | :-: | :-:| :-: | :-: |
| **Unregularized (Original Features)**  | 1.932715 | 0.564595 | 2.416220 | 0.130706  | 006.669397 |
| **Regularized (Original Features)**    | 1.889901 | 0.572407 | 2.276397 | 0.129983  | 476.689539 |
| **Regularized (Transformed Features)** | 3.869358 | 0.297641 | 5.153465 | 0.068069  | 2100.70408 |

<br>

| Dataset | Train Accuracy | Test Accuracy | Average Precision | Average Recall | Total Support | Epochs | Time Taken (s)
| :-: | :-: | :-: | :-:| :-: | :-: | :-: | :-: |
| [Olivetti Faces Dataset](http://scikit-learn.org/stable/datasets/olivetti_faces.html) | 94.393 | 95.938 | 96 | 97 | 96 | 320 | 46 | 96.859

The classification report containing average precision, recall, f1-score and total support along with the precision, recall, f1-score and support for each class is given in
the file 'classification_report.csv'. <br>
Optimal batch size was found to be 128. <br>

## Issues
At this point in time, there are some issues with the Windows version of
Auto-Keras given that you use 'CUDA' device instead of 'CPU'. It was 
experienced that the multiprocessing API and PyTorch do not work well together. The issues
are listed as follows: <br>

* "Traceback (most recent call last):
<br>... <br>
accuracy, loss, graph = train_results.get()[0] <br>
File "C:\Users\user\AppData\Local\Programs\Python\Python36\
\lib\multiprocessing\pool.py", line 644, in get
raise self._value <br>
TypeError: 'float' object cannot be interpreted as an integer". <br>
 Don't be mislead into thinking that the problem is related
to the multiprocessing API, it simply refers to explicitly casting values in the tuple
'self.padding' as int (C:\Users\user\AppData\Local\Programs\Python\Python36\site-packages\torch\nn\modules\conv.py at 
line 301 before calling the function F.conv2d with appropriate parameters)

* After fixing the former error, when you run the 'auto_keras_magic.py' script again, the program will fail with the following information
"Traceback (most recent call last):
<br>... <br>
THCudaCheck FAIL file=c:\new-builder_3\win-wheel\pytorch\torch\csrc\generic\StorageSharing.cpp line 231 error 71: operation not supported 
<br> ... <br>
File "C:\Users\user\AppData\Local\Programs\Python\Python36\lib\site-packages\autokeras\search.py", line 190 in search <br>
accuracy, loss, graph = train_results.get()[0] <br>
File "C:\Users\user\AppData\Local\Programs\Python\Python36\
\lib\multiprocessing\pool.py", line 644, in get
raise self._value <br>
multiprocessing.pool.MaybeEncodingError: error sending result '[(98.08, tensor=(2.3784, device='cuda:0'), <autokeras.graph.Graph.object at 0x000002821B58E668>)]' <br>
Reason: 'RuntimeError('cuda runtime error (71) : operation not supported at c:\\new-builder_3\\win-wheel\\pytorch\\torch\\csrc\\generic\\StorageSharing.cpp:231',)' <br>
Unfortunately it looks like multiprocessing and PyTorch do not seem to work well together. A hack to this problem is to replace line 178: <br>
"train_results = pool.map_async(train, [(graph, train_data, test_data, self.trainer_args, <br>
                                                os.path.join(self.path, str(model_id) + '.png'), self.verbose)])" <br>
with: <br>
train_results = train((graph, train_data, test_data, self.trainer_args, os.path.join(self.path, str(model_id) + '.png'), self.verbose)) <br>
and replace line 190 <br>
accuracy, loss, graph = train_results.get()[0] <br>
with: <br>
accuracy, loss, graph = train_results <br>
