import os

os.environ["COMET_DISABLE_AUTO_LOGGING"] = "1"

import comet_ml
exp = comet_ml.start()

import css
import logging
import sys

logging.basicConfig(
    format="%(asctime)s %(levelname)s -- %(name)s: %(message)s",
    level=logging.INFO,
    stream=sys.stdout,
)
import pandas as pd

all_data = pd.read_csv("data/usage_metrics.csv", index_col=0)
_product_data = pd.read_csv("data/product_metrics.csv", index_col=0)
_peer_dims_data = pd.read_csv("data/peer_dims.csv", index_col=0)

all_data[_product_data.columns] = _product_data
all_data[_peer_dims_data.columns] = _peer_dims_data
from css import config

service_model_config = config.ConfigModel.from_yaml("config.yaml")
service_model = service_model_config.to_obj()

print("Service model", service_model, type(service_model))

for children_key, children_value in service_model.children.items():
	print("Children", children_value, type(children_value))

	for grandchildren_key, grandchildren_value in children_value.children.items():
		print("Grandchildren", grandchildren_value, type(grandchildren_value))

result = service_model.fit(all_data)

print("Result", result, type(result))

# After training the model on all accounts
# Load or extract data for a single account
single_account_data = all_data.loc[["6ef13f1ddf7"]]  # Example using a specific account ID

# Score just that account
single_account_score = service_model.score(single_account_data)
print("Single account score:", single_account_score)

exp.log_table("final_score.csv", single_account_score)