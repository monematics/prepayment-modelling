#!/usr/bin/env python
# coding: utf-8

# In[78]:


import numpy as np
import pandas as pd
import csv
import glob
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.metrics import     r2_score, get_scorer
from sklearn.linear_model import     Lasso, Ridge, LassoCV,LinearRegression
from sklearn.preprocessing import     StandardScaler, PolynomialFeatures
from sklearn.model_selection import     KFold, RepeatedKFold, GridSearchCV,     cross_validate, train_test_split


# In[79]:


data = pd.read_csv("final_data_mbs.csv")


# In[80]:


data


# In[81]:


y = data[0:10000]["cpr"]
x = data.iloc[0:10000,1:-1]


# In[82]:


x


# In[83]:


y


# In[84]:


x_train , x_test, y_train, y_test = train_test_split(x, y, test_size=2000, random_state= 50)


# In[85]:


poly = PolynomialFeatures(
    degree = 2, include_bias = False, interaction_only = False)


# In[86]:


x_train_poly = poly.fit_transform(x_train)
polynomial_column_names =     poly.get_feature_names(input_features = x_train.columns)
x_train_poly =     pd.DataFrame(data = x_train_poly, 
        columns = polynomial_column_names )
x_train_poly.columns = x_train_poly.columns.str.replace(' ', '_')
x_train_poly.columns = x_train_poly.columns.str.replace('^', '_')


# In[87]:


sc = StandardScaler()
x_train_poly_scaled = sc.fit_transform(x_train_poly)
x_train_poly_scaled = pd.DataFrame(         data = x_train_poly_scaled, columns = x_train_poly.columns)


# In[88]:


x_test_poly = poly.transform(x_test)
x_test_poly_scaled = sc.transform(x_test_poly)


# In[89]:


cv = KFold(n_splits=5, shuffle=True, random_state= 50)


# In[90]:


lasso_alphas = np.linspace(0, 0.02, 11)


# In[91]:


def regmodel_param_plot(
    validation_score, train_score, alphas_to_try, chosen_alpha,
    scoring, model_name, test_score = None, filename = None):
    
    plt.figure(figsize = (8,8))
    sns.lineplot(y = validation_score, x = alphas_to_try, 
                 label = 'validation_data')
    sns.lineplot(y = train_score, x = alphas_to_try, 
                 label = 'training_data')
    plt.axvline(x=chosen_alpha, linestyle='--')
    if test_score is not None:
        sns.lineplot(y = test_score, x = alphas_to_try, 
                     label = 'test_data')
    plt.xlabel('alpha_parameter')
    plt.ylabel(scoring)
    plt.title(model_name + ' Regularisation')
    plt.legend()
    if filename is not None:
        plt.savefig(str(filename) + ".png")
    plt.show()


# In[92]:


def regmodel_param_test(
    alphas_to_try, x, y, cv, scoring = 'r2', 
    model_name = 'LASSO', x_test = None, y_test = None, 
    draw_plot = False, filename = None):
    
    validation_scores = []
    train_scores = []
    results_list = []
    if x_test is not None:
        test_scores = []
        scorer = get_scorer(scoring)
    else:
        test_scores = None

    for curr_alpha in alphas_to_try:
        
        if model_name == 'LASSO':
            regmodel = Lasso(alpha = curr_alpha)
        elif model_name == 'Ridge':
            regmodel = Ridge(alpha = curr_alpha)
        else:
            return None

        results = cross_validate(
            regmodel, x, y, scoring=scoring, cv=cv, 
            return_train_score = True)

        validation_scores.append(np.mean(results['test_score']))
        train_scores.append(np.mean(results['train_score']))
        results_list.append(results)

        if x_test is not None:
            regmodel.fit(x,y)
            y_pred = regmodel.predict(x_test)
            test_scores.append(scorer(regmodel, x_test, y_test))
    
    chosen_alpha_id = np.argmax(validation_scores)
    chosen_alpha = alphas_to_try[chosen_alpha_id]
    max_validation_score = np.max(validation_scores)
    if x_test is not None:
        test_score_at_chosen_alpha = test_scores[chosen_alpha_id]
    else:
        test_score_at_chosen_alpha = None
        
    if draw_plot:
        regmodel_param_plot(
            validation_scores, train_scores, alphas_to_try, chosen_alpha, 
            scoring, model_name, test_scores, filename)
    
    return chosen_alpha, max_validation_score, test_score_at_chosen_alpha


# In[93]:


chosen_alpha, max_validation_score, test_score_at_chosen_alpha =     regmodel_param_test(
        lasso_alphas, x_train_poly_scaled, y_train, 
        cv, scoring = 'r2', model_name = 'LASSO', 
        x_test = x_test_poly_scaled, y_test = y_test, 
        draw_plot = True, filename = 'lasso_wide_search')
print("Chosen alpha: %.5f" %     chosen_alpha)
print("Validation score: %.5f" %     max_validation_score)
print("Test score at chosen alpha: %.5f" %     test_score_at_chosen_alpha)


# In[94]:


import numpy as np
from sklearn.linear_model import LinearRegression


# In[95]:


model = LinearRegression()


# In[96]:


model.fit(x, y)


# In[97]:


model = LinearRegression().fit(x, y)


# In[98]:


r_sq = model.score(x, y)


# In[99]:


print(f"coefficient of determination: {r_sq}")


# In[100]:


print(f"intercept: {model.intercept_}")


# In[101]:


print(f"slope: {model.coef_}")


# In[102]:


y_pred = model.predict(x)
print(f"predicted response:\n{y_pred}")


# In[115]:


fig, ax = plt.subplots()
plot_x = np.linspace(0,10000,10000)
ax.plot(plot_x, y, 'o', label='data')
ax.plot(plot_x, y_pred, 'x', label='data')


# In[119]:


x_new = x = data.iloc[10000:20000,1:-1]


# In[120]:


y_new = model.predict(x_new)


# In[128]:


fig, ax = plt.subplots()

ax.plot(plot_x, y, 'o', label='data')
ax.plot(plot_x, y_new, 'x', label='data')


# In[129]:


from sklearn import metrics


# In[132]:


metrics.mean_absolute_error(data[10000:20000]["cpr"],y_new)


# In[133]:


metrics.mean_absolute_error(y,y_pred)


# In[134]:


metrics.mean_squared_error(data[10000:20000]["cpr"],y_new)


# In[135]:


metrics.mean_squared_error(y,y_pred)


# In[ ]:




