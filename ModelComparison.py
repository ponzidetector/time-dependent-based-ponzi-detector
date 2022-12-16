from sklearn.preprocessing import StandardScaler

from ExperimentUtils import *
from tensorflow import random
from lightgbm import LGBMClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVR, SVC
import xgboost as xgb
from sklearn.gaussian_process import GaussianProcessRegressor, GaussianProcessClassifier
from sklearn.gaussian_process.kernels import Matern
from keras.saving.save import load_model
from keras import Model

# Experiment Setting
settings = {
    "random_seed": 22,
    "over_sampling_rate": 0.5,
    "under_sampling_rate": 0.8,
    "k_fold_split": 5,
    "train_test_ratio": 0.2,
    "repeats": 500,
    "is_run_sampling": True,
    "is_filtered_non_zero_investment": False
}

nonPonzi_account_feature_path = os.path.join("features", "account", "NonPonziAccountFeatures.csv")
ponzi_account_feature_path = os.path.join("features", "account", "PonziAccountFeatures.csv")
nonPonzi_time_dependent_feature_path = os.path.join("features", "timedenpendent",
                                        "NonPonziTimeDependentFeatures.csv")
ponzi_time_dependent_feature_path = os.path.join("features", "timedenpendent",
                                         "PonziTimeDependentFeatures.csv")

chen_features = ["know_rate", "balance", "difference_idx", "paid_rate", "max_pay", "balance_rate", "payment_time"]
data_mining_features = ["nbr_tx_in", "nbr_tx_out", "total_investment", "total_reward", "avg_transfer_in_value",
                        "avg_transfer_out_value", "dev_transfer_in_value", "dev_transfer_out_value",
                        "avg_time_btw_tx", "lifetime", "gini_coefficient_to", "gini_coefficient_from",
                        "gini_time_in", "overlap_addr", "gini_time_out", "num_addr_out", "num_addr_in"]

# Random
np.random.seed(settings["random_seed"])
random.set_seed(settings["random_seed"])

metrics = {
    "Chen_baseline": init_metric_temp(),
    "Jung_baseline": init_metric_temp(),
    "ts_xgboost": init_metric_temp(),
    "ts_domain_xgboost": init_metric_temp(),
    "ts_knn": init_metric_temp(),
    "ts_domain_knn": init_metric_temp(),
    "ts_random_forest": init_metric_temp(),
    "ts_domain_random_forest": init_metric_temp(),
    "ts_svm": init_metric_temp(),
    "ts_domain_svm": init_metric_temp(),
    "ts_lightgbm": init_metric_temp(),
    "ts_domain_lightgbm": init_metric_temp(),
}

def run(X_train, X_test, y_train, y_test, metrics, model, isSTD=True):
    if isSTD:
        standard_scaler = StandardScaler()
        X_train = standard_scaler.fit_transform(X_train)  # standardizing the data
        X_test = standard_scaler.transform(X_test)
    a_trn, p_trn, r_trn, f1_trn, a_tst, p_tst, r_tst, f1_tst = model_running(X_train,
                                                                             X_test,
                                                                             y_train.values,
                                                                             y_test.values,
                                                                             model,
                                                                             settings["k_fold_split"])
    metric_recording(metrics, a_trn, p_trn, r_trn, f1_trn, a_tst, p_tst, r_tst, f1_tst)


def main():
    experiment_id, curves_path, data_splits_path, metrics_path, models_path = setup("output", settings)
    X, y = load_data(nonPonzi_account_feature_path, ponzi_account_feature_path, nonPonzi_time_dependent_feature_path,
                     ponzi_time_dependent_feature_path, settings["is_filtered_non_zero_investment"], seed = 100)

    for run_id in range(settings["repeats"]):
        seed = run_id + 1

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=settings["train_test_ratio"])
        train_test_split_saving(data_splits_path, run_id, X_train, X_test, y_train, y_test)

        # drop address:
        X_train = X_train.drop(["address"], axis=1)
        X_test = X_test.drop(["address"], axis=1)

        X_chen_train = X_train[chen_features]
        X_chen_test = X_test[chen_features]
        X_dmfd_train = X_train[data_mining_features]
        X_dmfd_test = X_test[data_mining_features]

        # Run Chen baseline
        chen_model = xgb.XGBClassifier(use_label_encoder=False, random_state=seed)
        run(X_chen_train, X_chen_test, y_train, y_test, metrics["chen_baseline"], chen_model)

        # Run DMFD baseline
        DMFD_model = RandomForestClassifier(random_state=seed)
        run(X_dmfd_train, X_dmfd_test, y_train, y_test, metrics["DMFD_baseline"], DMFD_model)

        # sampling
        if settings["is_run_sampling"]:
            data_sampling = random_sampling(X_train,
                                            y_train,
                                            0.5,
                                            0.8,
                                            seed=seed)
            X_train = data_sampling.drop(["label"], axis=1)
            y_train = data_sampling['label']

        X_chen_train = X_train[chen_features]
        X_chen_test = X_test[chen_features]
        X_dmfd_train = X_train[data_mining_features]
        X_dmfd_test = X_test[data_mining_features]

        seed = (run_id + 1) * 10
        # Run Chen baseline
        chen_model = xgb.XGBClassifier(use_label_encoder=False, random_state=seed)
        run(X_chen_train, X_chen_test, y_train, y_test, metrics["chen_baseline"], chen_model)

        # Run Jung baseline
        DMFD_model = RandomForestClassifier(random_state=seed)
        run(X_dmfd_train, X_dmfd_test, y_train, y_test, metrics["DMFD_baseline"], DMFD_model)

        X_ts_train = X_train.iloc[:, :516]
        X_ts_test = X_test.iloc[:, :516]

        # TS XGBoost
        ts_xgboost = xgb.XGBClassifier(use_label_encoder=False, random_state=seed)
        run(X_ts_train, X_ts_test, y_train, y_test, metrics["ts_xgboost"], ts_xgboost)

        # # TS Domain XGBoost
        ts_xgboost = xgb.XGBClassifier(use_label_encoder=False, random_state=seed)
        run(X_train, X_test, y_train, y_test, metrics["ts_domain_xgboost"], ts_xgboost)

        # TS KNN
        ts_knn = KNeighborsClassifier()
        run(X_ts_train, X_ts_test, y_train, y_test, metrics["ts_knn"], ts_knn)

        # # TS Domain KNN
        ts_domain_knn = KNeighborsClassifier()
        run(X_train, X_test, y_train, y_test, metrics["ts_domain_knn"], ts_domain_knn)

        # TS Random Forest
        ts_random_forest = RandomForestClassifier(random_state=seed)
        run(X_ts_train, X_ts_test, y_train, y_test, metrics["ts_random_forest"], ts_random_forest)

        # # TS Domain Random Forest
        ts_domain_random_forest = RandomForestClassifier(random_state=seed)
        run(X_train, X_test, y_train, y_test, metrics["ts_domain_random_forest"], ts_domain_random_forest)

        # TS SVM
        ts_svm = SVC(kernel='poly', C=1, degree=3)
        run(X_ts_train, X_ts_test, y_train, y_test, metrics["ts_svm"], ts_svm)

        # # TS Domain SVM
        ts_domain_svm = SVC(kernel='poly', C=1, degree=3)
        run(X_train, X_test, y_train, y_test, metrics["ts_domain_svm"], ts_domain_svm)

        # TS LightGBM
        ts_lightgbm = LGBMClassifier(random_state=seed)
        run(X_ts_train, X_ts_test, y_train, y_test, metrics["ts_lightgbm"], ts_lightgbm)

        # TS domain LightGBM
        ts_domain_lightgbm = LGBMClassifier(random_state=seed)
        run(X_train, X_test, y_train, y_test, metrics["ts_domain_lightgbm"], ts_domain_lightgbm)

    print("Exporting metrics records")
    save_metrics(metrics_path, metrics)

if __name__ == "__main__":
    main()

