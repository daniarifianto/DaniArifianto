import joblib

def predict(data):
    model = joblib.load("knn_model.sav")
    le = joblib.load("label_encoder.sav")
    pred = model.predict(data)
    return le.inverse_transform(pred)