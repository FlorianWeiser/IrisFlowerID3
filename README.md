ID3 Implementation for the Iris Flower Dataset
==============================================

ID3 is a simple algorithm invented by Ross Quinlan in 1986 to build decision trees based on the information gain
criterion and without pruning.
In this project, the ID3 algorithm was modified to perform binary splits and applied to the Iris flower dataset.

Dataset
-------
The [Iris flower dataset](https://archive.ics.uci.edu/ml/datasets/iris) consists of 50 samples from each of three species of Iris (Iris Setosa, Iris virginica, and Iris versicolor).
Each record lists the sepal_length, sepal_width, petal_length, petal_width and species.

|sepal_length|sepal_width|petal_length|petal_width|species    |
|------------|-----------|------------|-----------|-----------|
|5.1         |3.5        |1.4         |0.2        |Iris-setosa|
|4.9         |3          |1.4         |0.2        |Iris-setosa|
|4.7         |3.2        |1.3         |0.2        |Iris-setosa|
|4.6         |3.1        |1.5         |0.2        |Iris-setosa|
|...         |...        |...         |...        |...        |

Decision Tree Generation
------------------------

The ID3 algorithm starts with a single node and gradually performs binary splits so that the information gain is maximized.
Growing stops in this implementation, if all records in a leaf belong to the same Iris species, if the maximum tree depth is reached or if the number of samples in a leaf falls below the threshold.
See Python code comments for a more detailed explanation how the decision tree is built.

Output
------
The program outputs the generated binary tree and calculates its accuracy on the test set. As the training and test sets are selected randomly, the structure of the tree might differ for multiple program executions.
Potential output:
```
petal_width<1.0?
	[True] Iris-setosa
	[False] petal_width<1.8?
		[True] petal_length<5.0?
			[True] Iris-versicolor
			[False] Iris-virginica
		[False] Iris-virginica
accuracy on test set: 97.30%
```
