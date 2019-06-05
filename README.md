# Persuasion_Strategy

Modeling what makes a request persuasive—-eliciting the desired response from a reader--is critical to the study of propaganda, behavioral economics, and advertising. Yet current models can’t quantify the persuasiveness of requests or extract successful persuasive strategies. Building on theories of persuasion, we propose a neural network to quantify persuasiveness and identify the persuasive strategies in advocacy requests. Our semi-supervised hierarchical neural network model is supervised by the number of people persuaded to take actions and partially supervised at the sentence level with human-labeled rhetorical strategies. Our method outperforms several baselines, uncovers persuasive strategies—offering increased interpretability of persuasive speech and has applications for other situations with document-level supervision but only partial sentence supervision.

## Code Structure

* ./src/AttnModel.py contains the codes for models.    
* ./src/DataLoader.py contains the codes for data pre-processing and generate batches for training.     
* ./src/MessageLoss.py contains the loss functions.     
* ./src/train.py and test.py contain codes for training and testing.

## Run

Run ```python ./src/train.py``` to train and save the model.         
Run ```python ./src/test.py```  to test the model.

## Data

The data comes from Kiva.org. If you want to get the data for research, please contact chenjiaao1998@gmail.com.

## Cite
If you use our tools for your work, please cite the following paper:

* Diyi Yang, Jiaao Chen, Zichao Yang, Dan Jurafsky, Eduard Hovy. "Let’s Make Your Request More Persuasive: Modeling Persuasive Strategies via Semi-Supervised Neural Nets on Crowdfunding Platforms" In NAACL. 2019.

