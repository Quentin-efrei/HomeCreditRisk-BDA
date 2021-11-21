import shap  # package used to calculate Shap values
import pickle
import pandas as pd
from functions import scale_data
import matplotlib.pyplot as plt

filename = "models/xgboostclassifier.sav"
loaded_model = pickle.load(open(filename, 'rb'))
# Create object that can calculate shap values
explainer = shap.TreeExplainer(loaded_model)

df = pd.read_csv('ready-data/test.csv')
X = scale_data(df) # normalizing data before training
# calculate shap values. This is what we will plot.
# Calculate shap_values for all of val_X rather than a single row, to have more data for plot.
shap_values = explainer.shap_values(X)
print(shap_values)
# Make plot. Index of [1] is explained in text below.
fig1 = shap.summary_plot(shap_values,X,show=False)
plt.savefig('plot/shap_summary_plot_xgboost.png')

#fig2 = shap.plots.scatter(shap_values[:, "OWN_CAR_AGE"], color=shap_values)
fig2 = shap.dependence_plot("EXT_SOURCE_3", shap_values, X)
plt.savefig('plot/shap_ext_source_3_plot_xgboost.png')