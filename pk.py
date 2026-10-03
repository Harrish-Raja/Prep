import os
import joblib
import pandas as pd

from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import root_mean_squared_error, mean_absolute_error, r2_score


Test_path = 'data/cold_storage_train.csv'
Train_path = 'data/cold_storage_test.csv'
Predict_path = 'data/cold_storage_predict.csv'

Train_cleaned = 'processed_data/cold_storage_train_cleaned.csv'
Test_cleaned = 'processed_data/cold_storage_test_cleaned.csv'
Predict_cleaned = 'processed_data/cold_storage_predict_cleaned.csv'

Tree_path = 'artifacts/decision_tree_model.pkl'
Boost_path = 'artifacts/gradient_boosting_model.pkl'
Output_path = 'output/cold_storage_energy_predictions.csv'

target = 'electricity_consumed_kwh'


def clean_data(input_path, output_path):
    df = pd.read_csv(input_path)

    df = df.drop(columns=[
        'run_id',
        'operation_date',
        'facility_location',
        'chamber_code',
        'shift_supervisor',
        'client_contract_ref'
    ])

    df = df.drop_duplicates().reset_index(drop=True)

    for i in [
        'outside_temperature_celsius',
        'door_open_minutes',
        'compressor_age_months'
    ]:
        df[i] = df[i].fillna(df[i].median())

    df['storage_mode'] = df['storage_mode'].str.lower().str.strip().map({
        'frozen': 0,
        'chilled': 1,
        'controlled': 2,
        'mixed': 3
    })

    df['insulation_grade'] = df['insulation_grade'].str.lower().str.strip().map({
        'standard': 0,
        'enhanced': 1,
        'high-efficiency': 2
    })

    df['uses_rooftop_solar'] = df['uses_rooftop_solar'].str.lower().str.strip().map({
        'no': 0,
        'yes': 1
    })

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_csv(output_path, index=False)

    return df


def train_tree_model(processed_path, model_path):
    df = pd.read_csv(processed_path)

    X = df.drop(columns=[target])
    y = df[target]

    model = DecisionTreeRegressor(
        max_depth=10,
        random_state=42
    )

    model.fit(X, y)

    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    joblib.dump(model, model_path)

    return model


def train_boosting_model(processed_path, model_path):
    df = pd.read_csv(processed_path)

    X = df.drop(columns=[target])
    y = df[target]

    model = GradientBoostingRegressor(
        n_estimators=200,
        max_depth=3,
        learning_rate=0.1,
        random_state=42
    )

    model.fit(X, y)

    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    joblib.dump(model, model_path)

    return model


def compare_models(
    tree_model_path,
    boosting_model_path,
    features_test,
    target_test
):
    tree = joblib.load(tree_model_path)
    boost = joblib.load(boosting_model_path)

    tree_pred = tree.predict(features_test)
    boost_pred = boost.predict(features_test)

    result = pd.DataFrame([
        {
            'model_name': 'DecisionTreeRegressor',
            'rmse': root_mean_squared_error(target_test, tree_pred),
            'mae': mean_absolute_error(target_test, tree_pred),
            'r2_score': r2_score(target_test, tree_pred)
        },
        {
            'model_name': 'GradientBoostingRegressor',
            'rmse': root_mean_squared_error(target_test, boost_pred),
            'mae': mean_absolute_error(target_test, boost_pred),
            'r2_score': r2_score(target_test, boost_pred)
        }
    ])

    result = result.sort_values(
        'r2_score',
        ascending=False
    ).reset_index(drop=True)

    return result


def evaluate_model(model_path, features_test, target_test):
    model = joblib.load(model_path)

    pred = model.predict(features_test)

    rmse = root_mean_squared_error(target_test, pred)
    mae = mean_absolute_error(target_test, pred)
    r2 = r2_score(target_test, pred)

    return rmse, mae, r2


def predict_new_data(
    model_path,
    input_path,
    processed_path,
    output_path
):
    model = joblib.load(model_path)

    raw_df = pd.read_csv(input_path)

    ids = raw_df['run_id']

    clean_df = clean_data(input_path, processed_path)

    pred = model.predict(clean_df)

    result = pd.DataFrame({
        'run_id': ids,
        'predicted_electricity_consumed_kwh': pred
    })

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    result.to_csv(output_path, index=False)

    return result


if __name__ == '__main__':

    train = clean_data(Train_path, Train_cleaned)
    test = clean_data(Test_path, Test_cleaned)

    print('Cleaned training runs:', train.shape)
    print('Cleaned held-back runs:', test.shape)

    train_tree_model(Train_cleaned, Tree_path)
    train_boosting_model(Train_cleaned, Boost_path)

    print('Models trained and saved.')

    features = test.drop(columns=[target])
    target_test = test[target]

    comparison = compare_models(
        Tree_path,
        Boost_path,
        features,
        target_test
    )

    print('Model comparison:')
    print(comparison)

    best_model = comparison.loc[0, 'model_name']

    path = {
        'DecisionTreeRegressor': Tree_path,
        'GradientBoostingRegressor': Boost_path
    }

    best_path = path.get(best_model)

    rmse, mae, r2 = evaluate_model(
        best_path,
        features,
        target_test
    )

    print(
        f'RMSE: {rmse:.2f} | '
        f'MAE: {mae:.2f} | '
        f'R2: {r2:.4f}'
    )

    df = predict_new_data(
        best_path,
        Predict_path,
        Predict_cleaned,
        Output_path
    )

    print('Predictions written:', df.shape)