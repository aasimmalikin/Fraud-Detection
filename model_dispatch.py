from sklearn import linear_model, ensemble
models = {
    'logistic_regression' : linear_model.LogisticRegression(verbose= True, max_iter = 1000, random_state=10),
    'random_forest' : ensemble.RandomForestClassifier(verbose = True, n_estimators = 100, criterion = 'gini')

}