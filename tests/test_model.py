from ml_model.train import train_model
def test_train_model():
  # Train the model
  model = train_model()
  print(f"Model: {model}")  # Print the value of model
  assert model is not None # Ensure the model is trained

def test_predict():
    #test if predictiion is working
    from ml_model.predict import predict
    sample_input = [15, 39]
    cluster = predict(sample_input)
    assert cluster == 4 # Ensure the prediction is correct
    
