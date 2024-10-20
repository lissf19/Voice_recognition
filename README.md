The main dataset used in this project and used by us to compare teams performances is DAPS
(Device and Produced Speech) Dataset (https://zenodo.org/records/4660670 ).
Six speakers from this dataset: F1, F7, F8, M3, M6, M8, form Class 1. The other 14 speakers belong to Class 0.

As a final user:
1. I am initializing the listening procedure.
2. The program listens to me with the use of a laptop microphone and detects the class.
3. Detected class and, optionally, confidence level, is displayed on the screen. The Jupyter-notebook program is used in this scenario.

The chosen metrics for evaluation - f1-score. 
The chosen models: Neural - CNN, MLP, Classical - Random Forest Classifier together with RandomizedSearchCV applied. 

Main CNN model: 
The chosen activation function - Relu
Splitting - 80% train, 20% validation <=> needs to be changed to 70-20-10 
Optimizer - Adam
Scheduler - StepLR (step_size=5, gamma=0.1)
Noise Reduction - noisereduse Fast Fourier transform (FFT)


