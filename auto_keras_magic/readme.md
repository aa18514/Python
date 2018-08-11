# Machine Learning Using AutoKeras

## Description
Use of an open source library [AutoKeras](https://autokeras.com/) for automated machine learning.
AutoKeras provides the functions to automatically search for architecture
and parameters for deep learning models.

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
TypeError: 'float' object cannot be interpreted as an integer". One possible solution to the problem is to explicitly cast values in the tuple
'self.padding' as int (C:\Users\user\AppData\Local\Programs\Python\Python36\site-packages\torch\nn\modules\conv.py at 
line 301 before calling the function F.conv2d with appropriate parameters)

* After fixing the former error, when you run the 'olivetti_faces.py' script again, the program will fail with the following information
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
Reason: 'RuntimeError('cuda runtime error (71) : operation not supported at c:\\new-builder_3\\win-wheel\\pytorch\\torch\\csrc\\generic\\StorageSharing.cpp:231',)'