from datetime import datetime
import numpy as np
import scipy.stats as stats
import pandas as pd

pd.options.mode.chained_assignment = None  # default='warn'

# Constants
day_in_epoch = 24 * 60 * 60

wei_to_eth_ratio = 1000000000000000000


def data_preprocess(txs):
    txs = txs[txs['isError'] == 'None']
    # parse wei to ether
    txs['value'] = txs['value'].astype(np.longdouble) / wei_to_eth_ratio
    return txs


def get_in_txs(address, txs):
    in_txs = txs.loc[txs["to"] == address]
    return in_txs[in_txs['isError'] == 'None']


def get_out_txs(address, txs):
    in_txs = txs.loc[txs["from"] == address]
    return in_txs[in_txs['isError'] == 'None']


def sum_by_address(txs, group_by, target):
    sub_txs = txs[[group_by, target]]
    sub_txs[target] = sub_txs[target].astype(np.longdouble)
    sum_array = sub_txs.groupby(group_by)[target].sum().values.astype(np.longdouble)
    return sum_array


def gini(array):
    """Calculate the Gini coefficient of a numpy array."""
    # based on bottom eq:
    # http://www.statsdirect.com/help/generatedimages/equations/equation154.svg
    # from:
    # http://www.statsdirect.com/help/default.htm#nonparametric_methods/gini.htm
    # All values are treated equally, arrays must be 1d:
    # Source: https://github.com/oliviaguest/gini
    #
    if len(array) == 0:
        return np.Inf
    array = array.flatten()
    if np.amin(array) < 0:
        # Values cannot be negative:
        array -= np.amin(array)
    # Values cannot be 0:
    array += 0.0000001
    # Values must be sorted:
    array = np.sort(array)
    # Index per array element:
    index = np.arange(1, array.shape[0] + 1)
    # Number of array elements:
    n = len(array)
    # Gini coefficient:
    return (np.sum((2 * index - n - 1) * array)) / (n * np.sum(array))


def inv_pay_skewness(txs, address):
    participants = list()
    investment = {}
    payment = {}
    for idx, row in txs.iterrows():
        if row['to'] == address and row['from'] not in participants:
            participants.append(row['from'])
            if row['from'] in investment.keys():
                count = investment[row['from']]
                investment[row['from']] = count + 1
            else:
                investment[row['from']] = 1
        elif row['from'] == address and row['to'] not in participants:
            participants.append(row['to'])
            if row['to'] in payment.keys():
                count = payment[row['to']]
                payment[row['to']] = count + 1
            else:
                payment[row['to']] = 1
    v = []
    for account in participants:
        m = n = 0
        if account in investment:
            m = investment[account]
        if account in payment:
            n = payment[account]
        v.append(n - m)
    return 0 if len(v) < 3 else stats.skew(np.array(v))


def lifetime(txs):
    timestamp = txs['timestamp'].values
    first_tx_timestamp = timestamp[0]
    last_tx_timestamp = timestamp[-1]
    return last_tx_timestamp - first_tx_timestamp


def gini_amount_to(in_txs):
    sum_arr = sum_by_address(in_txs, "from", "value")
    return gini(sum_arr)


def gini_amount_from(out_txs):
    sum_arr = sum_by_address(out_txs, "to", "value")
    return gini(sum_arr)


def avg_transfer_value(txs):
    return txs["value"].mean()


def dev_transfer_value(txs):
    return txs["value"].std()


def avg_time_btw_tx(txs):
    time = []
    timestamp = txs['timestamp'].values
    for i in range(0, len(timestamp) - 1):
        tx = timestamp[i]
        tx_next = timestamp[i + 1]
        time.append(tx - tx_next)
    return 0 if len(time) == 0 else np.array(time).mean()


def paid_and_rewarded_address(in_txs, out_txs):
    rewarded = out_txs["to"].unique()
    rewarded_investors = in_txs.loc[in_txs['from'].isin(rewarded)]
    return len(rewarded_investors["from"].unique())


def gini_by_appear_frequency_in(in_txs):
    invest_arr = in_txs.groupby("from").count().values.astype(np.float64)
    return gini(invest_arr)


def gini_by_appear_frequency_out(out_txs):
    reward_arr = out_txs.groupby("to").count().values.astype(np.float64)
    return gini(reward_arr)


def max_count_reward(out_txs):
    reward_arr = out_txs.groupby("to").count().values.astype(np.float64)
    return 0 if len(reward_arr) == 0 else np.max(reward_arr)


def know_rate(txs, address, num_receivers):
    know_accounts = set()
    number_known_account_receive_payment = 0
    for idx, row in txs.iterrows():
        if row['to'] == address:
            know_accounts.add(row['from'])
        elif row['from'] == address and row['to'] in know_accounts:
            number_known_account_receive_payment += 1
    return 0 if num_receivers == 0 else float(number_known_account_receive_payment / num_receivers)


def difference_idx(txs, address):
    participants = list()
    investment = {}
    payment = {}
    for idx, row in txs.iterrows():
        if row['to'] == address and row['from'] not in participants:
            participants.append(row['from'])
            if row['from'] in investment.keys():
                count = investment[row['from']]
                investment[row['from']] = count + 1
            else:
                investment[row['from']] = 1
        elif row['from'] == address and row['to'] not in participants:
            participants.append(row['to'])
            if row['to'] in payment.keys():
                count = payment[row['to']]
                payment[row['to']] = count + 1
            else:
                payment[row['to']] = 1
    v = []
    for account in participants:
        m = n = 0
        if account in investment:
            m = investment[account]
        if account in payment:
            n = payment[account]
        v.append(n - m)
    return 0 if len(v) < 3 else stats.skew(np.array(v))


class Feature:
    def __init__(self, address, txs, label):
        # basic properties
        txs = data_preprocess(txs)
        in_txs = get_in_txs(address, txs)  # transaction to target contract
        out_txs = get_out_txs(address, txs)  # transaction from target contract
        self.address = address  # Contract address
        self.label = label  # Ponzi class label
        self.total_inv_amt = in_txs["value"].astype(np.float64).sum()
        self.total_pay_amt = out_txs["value"].astype(np.float64).sum()
        self.num_all_tx = len(txs)
        self.num_in_tx = len(in_txs)
        self.num_out_tx = len(out_txs)
        self.num_inv_acc = len(in_txs["from"].unique())
        self.num_pay_acc = len(out_txs["to"].unique())
        if len(txs) > 0:
            self.lifetime = lifetime(
                txs)  # the first transaction to the address, and the date of the last transaction to/from the address
            self.gini_amt_in = gini_amount_to(in_txs)  # The Gini coefficient of the values transferred to the address.
            self.gini_amt_out = gini_amount_from(
                out_txs)  # The Gini coefficient of the values transferred from the address.
            self.avg_inv_amt = avg_transfer_value(in_txs)  # The average of the values transferred to the address.
            self.avg_pay_amt = avg_transfer_value(out_txs)  # The average of the values transferred from the address.
            self.dev_inv_amt = dev_transfer_value(
                in_txs)  # The standard deviation of the values transferred to the address.
            self.dev_pay_amt = dev_transfer_value(
                out_txs)  # The standard deviation of the values transferred from the address.
            self.avg_time_btw_txs = avg_time_btw_tx(txs)  # average amount of time between two transactions
            self.overlap_addr = paid_and_rewarded_address(in_txs,
                                                          out_txs)  # the number of addresses that paid to the contract and also were paid by it.
            self.gini_time_in = gini_by_appear_frequency_in(
                in_txs)  # the Gini coefficient computed over the time of the transactions paid the smart contracts;
            self.gini_time_out = gini_by_appear_frequency_out(
                out_txs)  # the Gini coefficient computed over the time of the transactions was paid by the smart contracts
            self.know_rate = know_rate(txs, address,
                                       self.num_pay_acc)  # the proportion of receivers who have invested before payment
            self.balance = self.total_inv_amt - self.total_pay_amt  # the balance of the smart contract
            self.difference_idx = difference_idx(txs,
                                                 address)  # this index is used to measure the difference of counts between payment and investment for all participants in a contract
            self.paid_rate = 0 if self.num_inv_acc == 0 else float(
                self.overlap_addr) / self.num_inv_acc  # the proportion of investors who received at least one payment.
            self.max_pay = max_count_reward(out_txs)  # the maximum of counts of payments to participants.
            self.balance_rate = 0 if self.total_inv_amt == 0 else float(
                self.balance) / self.total_inv_amt  # balance as a percentage of total investments
            self.pay_skewness = inv_pay_skewness(txs, address)

    def __str__(self):
        return str(self.__class__) + ": " + str(self.__dict__)
