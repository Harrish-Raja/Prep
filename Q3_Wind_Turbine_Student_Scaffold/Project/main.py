import os
import joblib
import pandas as pd

from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import root_mean_squared_error, mean_absolute_error, r2_score


Train_path = 'data/wind_turbine_train.csv'
Test_path = 'data/wind_turbine_test.csv'
Predict_path = 'data/wind_turbine_predict.csv'

Train_cleaned = 'processed_data/wind_turbine_train_cleaned.csv'
Test_cleaned = 'processed_data/wind_turbine_test_cleaned.csv'
Predict_cleaned = 'processed_data/wind_turbine_predict_cleaned.csv'

Svr_path = 'artifacts/svr_model.pkl'
Forest_path = 'artifacts/random_forest_model.pkl'

Output_path = 'output/wind_turbine_predictions.csv'


def clean_data(input_path, output_path):
    df = pd.read_csv(input_path)

    df = df.drop(columns=[
        'turbine_id',
        'inspection_date',
        'wind_farm',
        'service_region',
        'maintenance_vendor',
        'work_order_ref'
    ])

    df = df.drop_duplicates(keep='first')

    for i in [
        'operating_hours',
        'rated_capacity_mw',
        'gearbox_vibration_mm_s'
    ]:
        df[i] = df[i].fillna(df[i].median())

    df['drive_train_type'] = df['drive_train_type'].str.lower().str.strip().map({
        'geared': 0,
        'direct_drive': 1,
        'hybrid': 2
    })

    df['weather_exposure'] = df['weather_exposure'].str.lower().str.strip().map({
        'low': 0,
        'moderate': 1,
        'high': 2,
        'extreme': 3
    })

    df['offshore_installation'] = df['offshore_installation'].str.lower().str.strip().map({
        'no': 0,
        'yes': 1
    })

    os.makedirs(
        os.path.dirname(output_path),
        exist_ok=True
    )

    df.to_csv(
        output_path,
        index=False
    )

    return df


def train_svr_model(processed_path, model_path):
    df = pd.read_csv(processed_path)

    X = df.drop(
        columns=['annual_maintenance_cost_lakh']
    )

    y = df['annual_maintenance_cost_lakh']

    model = Pipeline([
        ('scaler', StandardScaler()),
        ('svr', SVR(
            kernel='rbf'
        ))
    ])

    model.fit(X, y)

    os.makedirs(
        os.path.dirname(model_path),
        exist_ok=True
    )

    joblib.dump(
        model,
        model_path
    )

    return model


def train_forest_model(processed_path, model_path):
    df = pd.read_csv(processed_path)

    X = df.drop(
        columns=['annual_maintenance_cost_lakh']
    )

    y = df['annual_maintenance_cost_lakh']

    model = RandomForestRegressor(
        n_estimators=100,
        max_depth=14,
        min_samples_leaf=15,
        random_state=42
    )

    model.fit(X, y)

    os.makedirs(
        os.path.dirname(model_path),
        exist_ok=True
    )

    joblib.dump(
        model,
        model_path
    )

    return model


def compare_models(
    svr_model_path,
    forest_model_path,
    features_test,
    target_test
):
    svr = joblib.load(svr_model_path)
    forest = joblib.load(forest_model_path)

    svr_pred = svr.predict(features_test)
    forest_pred = forest.predict(features_test)

    result = pd.DataFrame([
        {
            'model_name': 'SVR',
            'rmse': root_mean_squared_error(
                target_test,
                svr_pred
            ),
            'mae': mean_absolute_error(
                target_test,
                svr_pred
            ),
            'r2_score': r2_score(
                target_test,
                svr_pred
            )
        },
        {
            'model_name': 'RandomForestRegressor',
            'rmse': root_mean_squared_error(
                target_test,
                forest_pred
            ),
            'mae': mean_absolute_error(
                target_test,
                forest_pred
            ),
            'r2_score': r2_score(
                target_test,
                forest_pred
            )
        }
    ])

    result = result.sort_values(
        'r2_score',
        ascending=False
    ).reset_index(drop=True)

    return result


def evaluate_model(
    model_path,
    features_test,
    target_test
):
    model = joblib.load(model_path)

    pred = model.predict(features_test)

    rmse = root_mean_squared_error(
        target_test,
        pred
    )

    mae = mean_absolute_error(
        target_test,
        pred
    )

    r2 = r2_score(
        target_test,
        pred
    )

    return rmse, mae, r2


def predict_new_data(
    model_path,
    input_path,
    processed_path,
    output_path
):
    model = joblib.load(model_path)

    raw_df = pd.read_csv(input_path)

    ids = raw_df['turbine_id']

    clean_df = clean_data(
        input_path,
        processed_path
    )

    pred = model.predict(clean_df)

    result = pd.DataFrame({
        'turbine_id': ids,
        'predicted_annual_maintenance_cost_lakh': pred
    })

    os.makedirs(
        os.path.dirname(output_path),
        exist_ok=True
    )

    result.to_csv(
        output_path,
        index=False
    )

    return result


if __name__ == '__main__':

    train = clean_data(
        Train_path,
        Train_cleaned
    )

    test = clean_data(
        Test_path,
        Test_cleaned
    )

    print(train.size)
    print(test.size)

    train_svr_model(
        Train_cleaned,
        Svr_path
    )

    train_forest_model(
        Train_cleaned,
        Forest_path
    )

    print('Models trained and saved.')

    features = test.drop(
        columns=['annual_maintenance_cost_lakh']
    )

    target = test['annual_maintenance_cost_lakh']

    comparison = compare_models(
        Svr_path,
        Forest_path,
        features,
        target
    )

    print(comparison)

    best_model = comparison.loc[
        0,
        'model_name'
    ]

    path = {
        'SVR': Svr_path,
        'RandomForestRegressor': Forest_path
    }

    best_path = path.get(best_model)

    a, b, c = evaluate_model(
        best_path,
        features,
        target
    )

    print(f'rmse:{a:.4f}')
    print(f'mae:{b:.4f}')
    print(f'r2_score:{c:.4f}')

    df = predict_new_data(
        best_path,
        Predict_path,
        Predict_cleaned,
        Output_path
    )

    print(df.shape)
