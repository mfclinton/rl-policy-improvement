This project is written entirely in Python
The code is located within the JupyterNotebooks (*.ipynb files) and the /lib/ directory


Required Python Libraries
- numpy
- torch (only necessary for old version)
- scipy
- cma (https://github.com/CMA-ES/pycma)
- IPython

In addition, you also need JupyterNotebook to launch and run the notebooks.
pip install any of the above libraries if you don't already have it.





Directions To Generate Policies
0. Download the data.csv (from project pdf) and place it in the "data" folder. (if you want to use the 687 Gridworld or Mockworld, use their respective Jupyter Notebooks to generate the data)
1. Install the above libraries
2. Launch JupyterNotebook (cmd -> jupyter notebook)
3. Navigate to the SynthesizePolicies.ipynb notebook
4. In the 2nd cell, labeled with "SET PARAMS" set the parameters according to the guide below.
	- USE_GRIDWORLD : If set to True it, policies are generated for the cs687 Gridworld. Otherwise, policies are generated for the project world.
	- USE_PDIS : If set to True we use Per Decision Importance Sampling for our policy improvement. Otherwise we use regular Importance Sampling.
	- percent_increase : The percentage we want to achieve over target performance. See /lib/DataManager.GetTargetPerformance for information about what our target performance is.
	- num_policies : Number of policies to generate
	- es_path : Path of the saved CMA-ES object (/pickles/ Gridworld or Vanilla *from given data*) that we load in and generate policies with (see Generating CMA-ES Object for more detail)
	- gamma : The gamma value used for our environment
	- delta : confidence threshold
5. Run the notebook. Note: that each policy is cross checked with Mockworld and the IS/PDIS bound and tossed out if it doesn't meet the specifications. If you want policies fast without precautions then comment out those lines in the generation loop.


Generating CMA-ES Object *see NOTE below for more info*
1. Navigate to Evaluation.ipynb
2. In the second cell, labeled "SET PARAMS" set the associated parameters as described below.
	- USE_GRIDWORLD : *see above*
	- USE_PDIS : *see above*
	- num_train_interval : How many intervals to train for our CMA-ES object
	- percent_increase : *see above*
	- gamma : *see above*
	- delta : *see above*



NOTE : Pretrained CMA-ES objects are available in /pickles/. but if you want to train your own follow the Generating CMA-ES Object guide.

CMA-ES Objects Used : "pickles\van\saved-cma-van-is-80.pkl" for 50% of policies, "pickles\van\saved-cma-PDIS-20.pkl" for 50% of policies
Parameters Used For Policy Generation
- for CMA-ES = "pickles\van\saved-cma-van-is-80.pkl" set USE_PDIS to FALSE
= for CMA-ES = "pickles\van\saved-cma-PDIS-20.pkl" set USE_PDIS to TRUE
- percent_increase = 0.1
- num_policies = 50 for each CMA-ES object
- es_path = path to respective CMA-ES object above
- gamma = 0.95
- delta = 0.01





~~~ Directory Info ~~~
Note : If a directory listed here (like outcmaes) is not included in the submitted zip, then it exists in my actual development environment but was excluded from the submission (likely because I deemed it irrelevant or taking up too much space).

-- Directory Explanation --
data : "Holds all csv files. From generated episodes of one of my worlds, to the data provided for the assignment."
figures : "Holds important images extracted from an analysis of the data"
lib : "Library containing important functions and classes used across notebooks"
outcmaes : "Log files associated with CMA module."
pickles : "Stores the trained CMA-ES objects."
policies : "Stores the numpy version of the policies."
text_policies : "Stores the text version of the policies."

-- Notebook Explanation --
cs687_gridworld : "This is my python implementation of the CS687 Gridworld from the HWs. This was used for testing purposes."
DataExploration.ipynb : "This notebook was used to explore the data and help create all figures in /figures/"
Evaluation : "This notebook was used for training a CMA-ES object to generate new policies"
Mock_Exploration : "Basically a clone of DataExploration but for analyzing the generated data of the Mockworld. Verifies it maintains statistics of original data."
Mockword : "Runs a policy on my best recreation of the environment from the provided data."
SynthesizePolicies : "This notebook loads a CMA-ES object and proceeds to generate, evaluate, and save policies."
CheckCreatedPolicies : "This notebook just double checks generated policies. Mostly identical to SynthesizePolicies except it reads from a directory"