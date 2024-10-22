# test_model.py
from model import model, X_test, y_test
from sklearn.metrics import accuracy_score

def test_accuracy():
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    assert accuracy > 0.7, f"Test failed, accuracy too low: {accuracy}"

if __name__ == "__main__":
    test_accuracy()
    print("All tests passed!")
