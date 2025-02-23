**Description**
This project performs high confidence off-policy improvement. Meaning that from the results of running another policy, the code produces a new policy that will perform better than the existing policy with high confidence.

This project is written entirely in Python
The code is located within the JupyterNotebooks (*.ipynb files) and the /lib/ directory


**Required Python Libraries**
- numpy
- torch (only necessary for old version)
- scipy
- cma (https://github.com/CMA-ES/pycma)
- IPython

In addition, you also need JupyterNotebook to launch and run the notebooks.
pip install any of the above libraries if you don't already have it.





**Directions To Generate Policies**
0. Download the data.csv (from project pdf) and place it in the "data" folder. (if you want to use the 687 Gridworld or Mockworld, use their respective Jupyter Notebooks to generate the data)
1. Install the above libraries
2. Launch JupyterNotebook (cmd -> jupyter notebook)
3. Navigate to the SynthesizePolicies.ipynb notebook
4. In the 2nd cell, labeled with "SET PARAMS" set the parameters according to the guide below.
	- USE_GRIDWORLD : If set to True it, policies are generated for the cs687 Gridworld. Otherwise, policies are generated for the project world.
	- USE_PDIS : If set to True we use Per Decision Importance Sampling for our policy improvement. Otherwise we use regular Importance Sampling.
		- Set to True for pdis_0.pkl, note below that it doesn't matter what this is for other pkls
	- USE_BOTH_PDIS : If True it takes policies that pass both PDIS and IS bounds
		- Set to True for all but pdis_0.pkl
	- percent_increase : The percentage we want to achieve over target performance. See /lib/DataManager.GetTargetPerformance for information about what our target performance is.
	- num_policies : Number of policies to generate
		- 40 for is_0
		- 40 for is_1, followed by es.ask(), followed by 10 more iterations (using 50 should yield sufficiently equal results)
		- 20 for each pdis
	- es_path : Path of the saved CMA-ES object (/pickles/ Gridworld or Vanilla *from given data*) that we load in and generate policies with (see Generating CMA-ES Object for more detail)
	- gamma : The gamma value used for our environment
	- delta : confidence threshold
	- random.seed() : *not the seed for the cma-es objects, see notes for that* For synthesizing policies, I used seeds 508 (is_1.pkl), 1337 (is_0.pkl), 88885555 (pdis_1.pkl), 877496444 (pdis_0.pkl), 3125832 (pdis_2.pkl)
		- Note: CMA-ES objects are perma-seeded since they're saved to a file, the above seed is just for numpy and other random stuff.
		- NOTE: to get the seed use for generating the cma-es objects, load them in and check their ".opts" for the ".seed" parameter and pass in {"seed":seedhere} as a 3rd parameter to the cma.CMAEvolutionStrategy object located in Evaluation Jupyter notebook in the cell containing the function "inv_barrier_constrained_explore"
			for convenience, the seeds are 768362 (is_1.pkl), 552873 (is_0.pkl), 493998 (pdis_1.pkl), 595346 (pdis_0.pkl), 698993 (pdis_2.pkl)
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
	- seed : seed to use for the CMA-ES agent
		- see seed note above in section 4 for seeds for CMA-ES and seeds for Numpy



NOTE : Pretrained CMA-ES objects are available in /pickles/. but if you want to train your own follow the Generating CMA-ES Object guide.

CMA-ES Objects Used in Synthesis notebook (located in pickles directory) : (is_1.pkl for 20%), (is_0.pkl for 20%), (pdis_1.pkl for 20%), (pdis_0.pkl for 20%), (pdis_2.pkl for 20%)
Parameters Used For Policy Generation
- for CMA-ES = if "is" in the CMA-ES name set USE_PDIS to FALSE, else if "pdis" in CMA-ES name set USE_PDIS to TRUE
- set USE_BOTH_PDIS True for all CMA-ES objects except pdis_0.pkl
- percent_increase = 0.1
- title offset = arbitrary, I used the order listed in the header of this section (is_1, is_0, pdis_1, ...)
- num_policies = 20 for each CMA-ES object
- es_path = path to respective CMA-ES object above
- gamma = 0.95
- delta = 0.01





**~~~ Directory Info ~~~**
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
