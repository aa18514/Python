# validating against floating point addition under-flows

This is only a initial attempt, there are many potential directions that the project could be taken to. <br> 

For example, distributing the compute amongst several machines, and exploring under-flow for different floating point representations <br> 
Another possible direction is to use to the inspect and ast module in Python to retrieve all floating point ops from scripts and calculate errors due to underflow <br>
The current version, generates exponents (with base 10) for sizes ranging from 1 to 100,000. The following diagrams show the percentage and absolute errors against the sample size: <br> 

<p align="center"> 
	<img src="https://github.com/aa18514/Python/blob/master/scientific_computing/images/absolute_error.png" /> 
</p> 


<p align="center"> 
	<img src="https://github.com/aa18514/Python/blob/master/scientific_computing/images/percentage_error.png" /> 
</p> 