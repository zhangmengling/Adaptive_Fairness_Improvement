# import xgboost
import shap

# train an XGBoost model
X, y = shap.datasets.adult()
print("-->X", X)
print("-->y", y)
# model = xgboost.XGBRegressor().fit(X, y)

from sklearn.neural_network import MLPClassifier
model = MLPClassifier(solver='adam', activation='identity', max_iter=500, alpha=1e-5,
                                hidden_layer_sizes=(64, 32, 16, 8, 4),
                                random_state=1, verbose=True)
model.fit(X, y)

# explain the model's predictions using SHAP
# (same syntax works for LightGBM, CatBoost, scikit-learn, transformers, Spark, etc.)
# shap_values = shap.Explainer(model)
explainer = shap.KernelExplainer(model.predict, X)
shap_values = explainer.shap_values(X)
print("-->rf_shap_values", shap_values)
shap.summary_plot(shap_values, X)

# # visualize the first prediction's explanation
# shap.plots.waterfall(shap_values[0])

# from sklearn.ensemble import RandomForestRegressor
# rf = RandomForestRegressor(max_depth=6, random_state=0, n_estimators=10)
# rf.fit(X, y)
# print(rf.feature_importances_)
# import shap
# rf_shap_values = shap.KernelExplainer(rf.predict, X)
# shap_values = rf_shap_values.shap_values(X)
# print("-->rf_shap_values", shap_values)
# shap.summary_plot(shap_values, X)
