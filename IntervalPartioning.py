import pandas as pd
import numpy as np
import os
from TimeDependentFeatures import Feature
from tqdm import tqdm

hourly = 3600 * 48
daily = 86400  # 24 hours
weekly = daily * 7
monthly = daily * 30
yearly = weekly * 52

wei_to_eth_ratio = 1000000000000000000
chunk_size = 1000000

common_attr = ["timestamp",
               "transactionHash",
               "from",
               "to",
               "fromIsContract",
               "toIsContract",
               "value",
               "isError",
               "txType",
               "callingFunction"]


def combine(normal_txs, internal_txs):
    normal_txs = normal_txs[common_attr]
    internal_txs = internal_txs[common_attr]
    txs = pd.concat([normal_txs, internal_txs], ignore_index=True)
    return txs.sort_values(by=["timestamp"])


def init_segments(start, end, timestep):
    if start > end:
        return None
    n = int((end - start) / timestep) + 1
    return [[] for i in range(n)]


def segmentation(txs, segments, start_ts, timestep):
    for idx, row in txs.iterrows():
        ts = row['timestamp']
        distance = ts - start_ts
        chunk_idx = int(distance / timestep)
        segments[chunk_idx].append(row)
    return segments


def ts_processing(normal_transaction_path, internal_transaction_path, address, feature_path, timestep):
    # path initiate
    contract_internal_txs_path = os.path.join(internal_transaction_path, address + ".csv")
    contract_normal_txs_path = os.path.join(normal_transaction_path, address + ".csv")

    # Load internal transaction data
    internal_txs = pd.DataFrame(columns=common_attr)
    if os.path.isfile(contract_internal_txs_path):
        internal_txs = pd.read_csv(contract_internal_txs_path)
        # parse wei to ether
        internal_txs['value'] = internal_txs['value'].astype(np.longdouble) / wei_to_eth_ratio
        # filter out error tx
        internal_txs = internal_txs[internal_txs['isError'] == 'None']
        internal_txs['txType'] = 'internal'

    # Load normal transaction data
    normal_txs = pd.DataFrame(columns=common_attr)
    if os.path.isfile(contract_normal_txs_path):
        normal_txs = pd.read_csv(contract_normal_txs_path)
        # parse wei to ether
        normal_txs['value'] = normal_txs['value'].astype(np.longdouble) / wei_to_eth_ratio
        # filter out error tx
        normal_txs = normal_txs[normal_txs['isError'] == 'None']
        normal_txs['txType'] = 'normal'

    # merge all txs
    all_txs = combine(normal_txs, internal_txs)
    print("Number of txs:", len(all_txs))
    if (len(all_txs) < 2):
        print("Too few data to process - skip contract:", address)
        return

    # prepare timeslots
    start_ts = all_txs['timestamp'].values[0]
    end_ts = all_txs['timestamp'].values[-1]
    print("Contract start-time:", start_ts)
    print("Contract end-time:", end_ts)

    # start segmentation
    segments = init_segments(start_ts, end_ts, timestep)
    print(">> Start segmentation")
    segments = segmentation(all_txs, segments, start_ts, timestep)

    # TS feature aggregation
    features = []
    last_balance = 0
    for txs_list in tqdm(segments):
        f = Feature(address.lower(), txs_list, all_txs.columns, last_balance)
        last_balance = f.balance
        features.append(f)
        # print(vars(f))
    pd.DataFrame.from_records([vars(f) for f in features]).to_csv(feature_path, index=False)


def ts_creating(contracts, normal_transaction_path, internal_transaction_path, feature_path, labels):
    for idx, address in tqdm(enumerate(contracts)):
        address = address.lower()
        label = labels[idx]
        label_path = "ponzi" if label == 1 else "nonPonzi"
        contract_feature_path = os.path.join(feature_path, label_path, address + ".csv")
        print("Processing address:" + address)
        ts_processing(normal_transaction_path, internal_transaction_path, address, contract_feature_path,
                      hourly)


def run():
    # PONZI CONTRACTS
    # Path initiate
    feature_path = os.path.join("data", "timeseries")  # output ts path

    ponzi_contract_path = os.path.join("data", "Ponzi.csv")
    ponzi_normal_transaction_path = os.path.join("data", "transactions", "ponzi", "external")
    ponzi_internal_transaction_path = os.path.join("data", "transactions", "ponzi", "internal")

    # Load contracts
    ponzi_contracts = pd.read_csv(ponzi_contract_path)["address"].values
    # Feature aggregating
    ts_creating(contracts=ponzi_contracts,
                internal_transaction_path=ponzi_internal_transaction_path,
                normal_transaction_path=ponzi_normal_transaction_path,
                feature_path=feature_path,
                labels=[1] * len(ponzi_contracts))

    # DAPP CONTRACTS
    # Path initiate
    dapp_contract_path = os.path.join("data", "nonPonzi.csv")
    dapp_normal_transaction_path = os.path.join("data", "transactions", "nonPonzi", "external")
    dapp_internal_transaction_path = os.path.join("data", "transactions", "nonPonzi", "internal")

    # Load contracts
    dapp_contracts = pd.read_csv(dapp_contract_path)["address"].values
    # Feature aggregating
    ts_creating(contracts=dapp_contracts,
                internal_transaction_path=dapp_internal_transaction_path,
                normal_transaction_path=dapp_normal_transaction_path,
                feature_path=feature_path,
                labels=[0] * len(dapp_contracts))


if __name__ == '__main__':
    run()
