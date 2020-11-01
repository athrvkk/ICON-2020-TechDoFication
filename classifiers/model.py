from sklearn.metrics import accuracy_score, f1_score

# ----------------------------------------- ML Classifier -----------------------------------------
    
def ml_classifier_model(model, x_train, x_val, y_train, y_val):
    model.fit(x_train, y_train)
    results = model.predict(x_val)
    return model, accuracy_score(results, y_val), f1_score(results, y_val, average=None) 
