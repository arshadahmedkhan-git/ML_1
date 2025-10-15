import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import Lasso
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import cross_val_score, KFold
from xgboost import XGBRegressor
import lightgbm as lgb
from catboost import CatBoostRegressor


train = pd.read_csv("train.csv")
test = pd.read_csv("test.csv")

test_ID = test["Id"]
train.drop("Id", axis=1, inplace=True)
test.drop("Id", axis=1, inplace=True)

y = np.log1p(train["SalePrice"])
train.drop("SalePrice", axis=1, inplace=True)

all_data = pd.concat([train, test], axis=0)

for col in all_data:
    if all_data[col].dtype == "object":
        all_data[col] = all_data[col].fillna("None")
    else:
        all_data[col] = all_data[col].fillna(all_data[col].median())

all_data["TotalSF"] = all_data["TotalBsmtSF"] + all_data["1stFlrSF"] + all_data["2ndFlrSF"]
all_data["OverallQual_GrLivArea"] = all_data["OverallQual"] * all_data["GrLivArea"]
all_data["OverallQual_TotalSF"] = all_data["OverallQual"] * all_data["TotalSF"]

all_data = pd.get_dummies(all_data)

ntrain = train.shape[0]
X = all_data[:ntrain]
X_test = all_data[ntrain:]

models = {
    "XGB": XGBRegressor(
        n_estimators=2000, learning_rate=0.05, max_depth=3,
        subsample=0.7, colsample_bytree=0.7, random_state=42
    ),
    "LGBM": lgb.LGBMRegressor(
        n_estimators=2000, learning_rate=0.05, num_leaves=30,
        subsample=0.7, colsample_bytree=0.7, random_state=42
    ),
    "CatBoost": CatBoostRegressor(
        iterations=2000, learning_rate=0.05, depth=6,
        verbose=0, random_state=42
    ),
    "GBR": GradientBoostingRegressor(
        n_estimators=3000, learning_rate=0.05, max_depth=4, random_state=42
    ),
    "Lasso": Lasso(alpha=0.0005, random_state=42, max_iter=10000)
}

def rmsle_cv(model):
    kf = KFold(5, shuffle=True, random_state=42)
    rmse = np.sqrt(-cross_val_score(model, X, y, scoring="neg_mean_squared_error", cv=kf))
    return rmse.mean()

for name, model in models.items():
    score = rmsle_cv(model)
    print(f"{name}: {score:.5f}")

for model in models.values():
    model.fit(X, y)

preds = np.expm1(
    (models["XGB"].predict(X_test) +
     models["LGBM"].predict(X_test) +
     models["CatBoost"].predict(X_test) +
     models["GBR"].predict(X_test) +
     models["Lasso"].predict(X_test)) / 5
)

submission = pd.DataFrame({
    "Id": test_ID,
    "SalePrice": preds
})
submission.to_csv("optimized_submission.csv", index=False)

print(" Submission file created: optimized_submission.csv")# ML_1
regression techniques. Some parts would include cleaning of data, engineering of features, and ensemble modeling for the best submission to the Kaggle House Prices competition.
