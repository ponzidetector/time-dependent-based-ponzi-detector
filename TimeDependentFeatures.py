import numpy as np
import pandas as pd


def cat_txs(txs_list, address, columns):
    in_coming = []
    out_going = []
    internal_in_coming = []
    internal_out_going = []
    normal_in_coming = []
    normal_out_going = []
    for tx in txs_list:
        if tx['to'] == address:
            in_coming.append(tx)
            if tx['txType'] == 'internal':
                internal_in_coming.append(tx)
            else:
                normal_in_coming.append(tx)
        elif tx['from'] == address:
            out_going.append(tx)
            if tx['txType'] == 'internal':
                internal_out_going.append(tx)
            else:
                normal_out_going.append(tx)
    return pd.DataFrame(columns=columns, data=in_coming), \
           pd.DataFrame(columns=columns, data=out_going), \
           pd.DataFrame(columns=columns, data=internal_in_coming), \
           pd.DataFrame(columns=columns, data=internal_out_going), \
           pd.DataFrame(columns=columns, data=normal_in_coming), \
           pd.DataFrame(columns=columns, data=normal_out_going)


def count_unique_address(df):
    return len(np.unique(df[['from', 'to']].values))


def count_unique_calling_function(df):
    return len(df["callingFunction"].unique())


def contract_or_person(txs):
    contract_txs = txs.loc[(txs["fromIsContract"] == 1) & (txs["toIsContract"] == 1)]
    person_txs = txs.loc[((txs["fromIsContract"] == 1) & (txs["toIsContract"] == 0))
                         | ((txs["fromIsContract"] == 0) & (txs["toIsContract"] == 1))]
    return contract_txs, person_txs


class Feature:
    def __init__(self, address, txs_list, columns, balance=0):
        txs = pd.DataFrame(columns=columns, data=txs_list)
        in_coming, out_going, internal_in_coming, internal_out_going, normal_in_coming, normal_out_going = cat_txs(txs_list, address, columns)
        # ETH value
        self.profit_and_loss = in_coming["value"].sum() - out_going["value"].sum()
        self.balance = balance + self.profit_and_loss
        self.profit = in_coming["value"].sum()
        self.profit_from_internal_txs = internal_in_coming["value"].sum()
        self.profit_from_normal_txs = normal_in_coming["value"].sum()
        self.loss = out_going["value"].sum()
        self.loss_from_internal_txs = internal_out_going["value"].sum()
        self.loss_from_normal_txs = normal_out_going["value"].sum()
        # Txs
        self.total_txs = len(txs_list)
        self.total_in_coming_txs = len(in_coming)
        self.total_out_going_txs = len(out_going)
        self.total_in_coming_internal_txs = len(internal_in_coming)
        self.total_out_going_internal_txs = len(internal_out_going)
        self.total_in_coming_normal_txs = len(normal_in_coming)
        self.total_out_going_normal_txs = len(normal_out_going)
        self.total_internal_txs = self.total_in_coming_internal_txs + self.total_out_going_internal_txs
        self.total_normal_txs = self.total_in_coming_normal_txs + self.total_out_going_normal_txs

        # Addresses
        self.total_unique_addresses = count_unique_address(txs)
        self.total_unique_in_coming_addresses = count_unique_address(in_coming)
        self.total_unique_out_going_addresses = count_unique_address(out_going)
        self.total_unique_in_coming_addresses_from_internal = count_unique_address(internal_in_coming)
        self.total_unique_out_going_addresses_from_internal = count_unique_address(internal_out_going)
        self.total_unique_in_coming_addresses_from_normal = count_unique_address(normal_in_coming)
        self.total_unique_out_going_addresses_from_normal = count_unique_address(normal_out_going)
        self.total_unique_calling_function = count_unique_calling_function(txs)
        self.total_unique_in_coming_calling_function = count_unique_calling_function(in_coming)
        self.total_unique_out_going_calling_function = count_unique_calling_function(out_going)
        self.total_unique_in_coming_calling_function_from_internal = count_unique_calling_function(internal_in_coming)
        self.total_unique_out_going_calling_function_from_internal = count_unique_calling_function(internal_out_going)
        self.total_unique_in_coming_calling_function_from_normal = count_unique_calling_function(normal_in_coming)
        self.total_unique_out_going_calling_function_from_normal = count_unique_calling_function(normal_out_going)

        # Contract or Account
        in_coming_contract_txs, in_coming_person_txs = contract_or_person(in_coming)
        out_going_contract_txs, out_going_person_txs = contract_or_person(out_going)
        self.profit_by_contract = in_coming_contract_txs["value"].sum()
        self.loss_by_contract = out_going_contract_txs["value"].sum()
        self.profit_by_person = in_coming_person_txs["value"].sum()
        self.loss_by_person = out_going_person_txs["value"].sum()
        self.num_in_coming_txs_from_contract = len(in_coming_contract_txs)
        self.num_out_going_txs_to_contract = len(out_going_contract_txs)
        self.num_in_coming_txs_from_person = len(in_coming_person_txs)
        self.num_out_going_txs_to_person = len(out_going_person_txs)
        self.num_unique_in_coming_contract_address = count_unique_address(in_coming_contract_txs)
        self.num_unique_out_going_contract_address = count_unique_address(out_going_contract_txs)
        self.num_unique_in_coming_person_address = count_unique_address(in_coming_person_txs)
        self.num_unique_out_going_person_address = count_unique_address(out_going_person_txs)

        def __str__(self):
            return str(self.__class__) + ": " + str(self.__dict__)
