## Time-series list
Below is the list of 43 different time-series we used to represent the change of information associated with an application throughout its lifetime in different aspects| 
Those time-series were derived from basic transaction information|
We then grouped them by the information from which they were created| All of these data depend on time, e|g|, which day they were measured|

### ETH value

| Time-series  | Description |
| ------------- | ------------- |
| balance  | the amount of ETH in a contract  |
| profit_and_loss  | subtraction of total investments (profit) and total payments (loss) of a contract   |
| loss  | total ETH amounts that a contract pays to its participants  |
| loss_by_contract  | total ETH amount sent from a contract to other contracts  |
| loss_by_person  | total ETH amount sent from a contract to the other user accounts  |
| loss_from_internal_txs  | total ETH amount recorded by internal transactions that a contract pays to its participants  |
| loss_from_normal_txs  | total ETH amount recorded by external transactions that a contract pays to its participants |
| profit  | total ETH amount that the contract received from its participants  |
| profit_by_contract  | total ETH amount that the contract received from other contracts  |
| profit_by_person  | total ETH amount that the contract received from other user accounts  |
| profit_from_internal_txs  | total ETH amount recorded by internal transactions that a contract pays received from its participants  |
| profit_from_normal_txs  | total ETH amount recorded by external transactions that a contract pays received from its participants  |

### Transaction

| Time-series   | Description |
| ------------- | ------------- |
|  total_txs| total number of transactions|
|  total_internal_txs| total number of internal transactions|
|  total_in_coming_txs| total number of transactions sent to a contract|
|  total_in_coming_internal_txs| total number of internal transactions sent to the contract|
|  total_in_coming_normal_txs| total number of external transactions sent to the contract|
|  total_normal_txs| total number of external transactions|
|  total_out_going_txs| total number of transactions sent from a contract|
|  total_out_going_internal_txs| total number of internal transactions sent from a contract|
|  total_out_going_normal_txs| total number of external transactions sent from a contract|


### Paricipant address

| Time-series   | Description |
| ------------- | ------------- |
| total_unique_addresses| total number of distinct participants (addresses) of a contract|
| total_unique_in_coming_addresses| total number of distinct participants who sent transactions to a contract|
| total_unique_in_coming_addresses_from_internal| total number of distinct participants who sent internal transactions to a contract|
| total_unique_in_coming_addresses_from_normal|  total number of distinct participants who sent external transactions to a contract|
| total_unique_out_going_addresses| total number of distinct participants who receive transactions from a contract|
| total_unique_out_going_addresses_from_internal| total number of distinct participants who receive internal transactions from a contract|
| total_unique_out_going_addresses_from_normal| total number of distinct participants who receive external transactions from a contract|

### Calling function

| Time-series   | Description |
| ------------- | ------------- |
| total_unique_calling_function| total number of distinct functions were called by a contract or its participants|
| total_unique_in_coming_calling_function| total number of distinct functions called by participants|
| total_unique_in_coming_calling_function_from_internal| total number of distinct functions called via internal transactions by participants|
| total_unique_in_coming_calling_function_from_normal| total number of distinct functions called via external transactions by participants|
| total_unique_out_going_calling_function| total number of distinct functions called by contracts|
| total_unique_out_going_calling_function_from_internal|  total number of distinct functions called via internal transactions by contracts|
| total_unique_out_going_calling_function_from_normal| total number of distinct functions called via external transactions by contracts|

### Participant account type

| Time-series   | Description |
| ------------- | ------------- |
| num_in_coming_txs_from_contract| number of transactions sent to a contract from other contracts|
| num_in_coming_txs_from_person| number of transactions sent to a contract from other user accounts|
| num_out_going_txs_to_contract| number of transactions sent from a contract to other contracts|
| num_out_going_txs_to_person| number of transactions sent from a contract to other user accounts|
| num_unique_in_coming_contract_address| number of distinct contracts that sent transactions to a contract|
| num_unique_in_coming_person_address| number of distinct user accounts that sent transactions to a contract|
| num_unique_out_going_contract_address| number of distinct contracts that received transactions from a contract|
| num_unique_out_going_person_address| number of distinct user accounts that received transactions from a contract|

## Time-series statistical measures
Below are the 12 statistical measures that were used to capture the characteristics of a time-series|

|Measure | Description |
| ------------- | ------------- |
| Mean  | Mean value of intervals  |
| Var  |  Variance value of intervals  |
| ACF1  | First order of auto-correlation of the series  |
| Linearity  | Strength of linearity calculated based on the coefficients of an orthogonal quadratic regression  |
| Curvature  | Strength of curvature  calculated based on the coefficients of an orthogonal quadratic regression  |
| Trend  | Strength of trend of a time-series based on an STL decomposition  |
| Season  | Strength of seasonality of a time-series based on an STL  |
| Entropy  | Spectral entropy measures the “forecastability” of a time-series, where low values indicate a high signal-to-noise ratio, and large values occur when a series is difficult to forecast  |
| Lumpiness  | Changing variance in remainder computed on non-overlapping windows  |
| Spikiness  | Strength of spikiness which is variance of the leave-one-out variances of the remainder component  |
| Fspots | Flat spot using discretization  computed by dividing the sample space of a time-series into ten equal-sized intervals, and computing the maximum run length within any single interval  |
| Cpoints | The number of times a time-series crosses the mean line  |

