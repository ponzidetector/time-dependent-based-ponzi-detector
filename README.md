# Time-dependent Based Ponzi
Time-dependent Based Ponzi is an approach that uses transaction information to detect Ponzi applications on Ethereum. This repository contains source code for data processing and experiment
in the research **Improving Robustness and Accuracy of Ponzi Scheme Detection on Ethereum Using Time-Dependent Features**. 

To run experiment we need to follow steps below:

### 1. Data Collecting
  Ethereum transaction data should be downloaded from [XBLOCK](http://xblock.pro/xblock-eth.html) repository. 
  After download 2 datasets *Block Transaction* and *Internal Transaction*, 
  transaction should be filter data from 2 address lists [Ponzi](https://github.com/ponzidetector/time-dependent-based-ponzi-detector/blob/master/data/Ponzi.csv) 
  and [Non-Ponzi](https://github.com/ponzidetector/time-dependent-based-ponzi-detector/blob/master/data/nonPonzi.csv). 
  Then the data should be grouped by contract and saved into `project/data/transactions/`
### 2. Features Aggregation
#### 2.1 Account Features
Account features that proposed in the paper could created by running the main function of `AccountFeaturesCreator.py`. 
Note that transaction data should be available in 
`project/data/transactions/ponzi/external` and `project/data/transactions/ponzi/internal` for Ponzi instances
and 
`project/data/transactions/nonPonzi/external` and `project/data/transactions/nonPonzi/internal` for nonPonzi instances. The outputs will be saved in `project/features/account/`
#### 2.2 Time-dependent Features
To aggeregate time-dependent features. Transaction data should be partitioned by running `IntervalPartioning.py` and then created by executing `TimeDependentFeatureCreator.py`. The outputs will be saved in `project/features/timedependent/`
### 3. Experiment
After aggregating appropriate Account Features and Time Dependent Features, experiments could be run from the file `ModelComparison.py`. 
The experiment results will be exported in `project/output` folder.
### 4. Contact
*TBD*
