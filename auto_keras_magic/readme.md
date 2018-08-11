#Machine Learning Using AutoKeras

##Description
Use of an open source library 'AutoKeras' for automated machine learning.
AutoKeras provides the functions to automatically search for architecture
and parameters for deep l earning models.

##Issues
At this point in time, there are some issues with the Windows version of
Auto-Keras given that you use 'CUDA' device instead of 'CPU'. It was 
experienced that the multiprocessing API and PyTorch do not work well together. The issues
are listed as follows: <br>
* Traceback (most recent call last):
<br>... <br>
accuracy, loss, graph = train_results.get()[0] <br>
File "C:\Users\user\AppData\Local\Programs\Python\Python36\
\lib\multiprocessing\pool.py", line 644, in get
raise self._value <br>
TypeError: 'float' object cannot be interpreted as an integer <br>

One solution to the problem is to explicitly cast values in the tuple
'self.padding' as int (C:\Users\user\AppData\Local\Programs\Python\Python36\site-packages\torch\nn\modules\conv.py at 
line 301 before calling the function F.conv2d with the appropiate parameters)

