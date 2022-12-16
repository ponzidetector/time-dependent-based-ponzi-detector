import pandas as pd
import os
from AccountFeatures import Feature

common_cols = ["timestamp",
               "transactionHash",
               "from",
               "to",
               "fromIsContract",
               "toIsContract",
               "value",
               "isError",
               "callingFunction"]


def combine(normal_txs, internal_txs):
    normal_txs = normal_txs[common_cols]
    internal_txs = internal_txs[common_cols]
    all_txs = pd.concat([normal_txs, internal_txs], ignore_index=True)
    return all_txs.sort_values(by=["timestamp"])


def aggregate_with_feature_class(address, normal_bev_path, internal_bev_path, label):
    print("Starting aggregate feature for contract: ", address, "[", label, "]")
    # Load normal txs
    normal_txs = pd.DataFrame(columns=common_cols)
    if os.path.isfile(normal_bev_path):
        normal_txs = pd.read_csv(normal_bev_path)

    # Load normal txs
    internal_txs = pd.DataFrame(columns=common_cols)
    if os.path.isfile(internal_bev_path):
        internal_txs = pd.read_csv(internal_bev_path)

    # Merge and sort
    all_txs = combine(normal_txs, internal_txs)
    print("Total number of txs: ", len(all_txs))
    if len(all_txs) < 1:
        print("Too few txs to process -> skip contract:", address)
        return None
    # Feature creation
    return Feature(address=address.lower(), txs=all_txs, label=label)


def batch_processing(chunk_size, contracts, internal_txs_path, normal_txs_path, feature_path, labels):
    if len(labels) != len(contracts):
        print("Inconsistent Data")
        return None
    # For logging
    count = 1
    total = len(contracts)
    chunk = []
    # Run
    for idx, address in enumerate(contracts):
        address = address.lower()
        print("Processing ", count, "/", total, " contracts")
        count += 1
        internal_path = os.path.join(internal_txs_path, address + ".csv")
        normal_path = os.path.join(normal_txs_path, address + ".csv")
        feature = aggregate_with_feature_class(address, internal_path, normal_path, labels[idx])
        if feature is None:
            continue
        print("Finished process feature for contract:", address)
        chunk.append(feature)
        if len(chunk) == chunk_size:
            save_df = pd.DataFrame.from_records([vars(f) for f in chunk])
            if os.path.isfile(feature_path):
                save_df.to_csv(feature_path, mode='a', header=False, index=False)
            else:
                save_df.to_csv(feature_path, index=False)
            chunk = []
    print("")


def run():
    # PONZI CONTRACT
    # Path initiate
    ponzi_contract_path = os.path.join("data", "Ponzi.csv")
    ponzi_normal_transaction_path = os.path.join("data", "transactions", "ponzi", "external")
    ponzi_internal_transaction_path = os.path.join("data", "transactions", "ponzi", "internal")
    ponzi_feature_path = os.path.join("features", "account", "PonziAccountFeatures.csv")

    # Load ponzi contracts
    ponzi_contracts = pd.read_csv(ponzi_contract_path)["address"].values


    # Batch processing
    batch_processing(chunk_size=20,
                     contracts=ponzi_contracts,
                     internal_txs_path=ponzi_internal_transaction_path,
                     normal_txs_path=ponzi_normal_transaction_path,
                     feature_path=ponzi_feature_path,
                     labels=[1] * len(ponzi_contracts))

    # DAPP CONTRACT
    # Path initiate
    dapp_contract_path = os.path.join("data", "nonPonzi.csv")
    dapp_normal_transaction_path = os.path.join("data", "transactions", "nonPonzi", "external")
    dapp_internal_transaction_path = os.path.join("data", "transactions", "nonPonzi", "internal")
    dapp_feature_path = os.path.join("features", "account", "NonPonziAccountFeatures.csv")

    # Load dapp contracts
    dapp_contracts = pd.read_csv(dapp_contract_path)["address"].values

    # Batch processing
    batch_processing(chunk_size=20,
                     contracts=dapp_contracts,
                     internal_txs_path=dapp_internal_transaction_path,
                     normal_txs_path=dapp_normal_transaction_path,
                     feature_path=dapp_feature_path,
                     labels=[0] * len(dapp_contracts))


if __name__ == "__main__":
    run()
