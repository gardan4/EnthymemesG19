# Load ground truth labels
with open('./D3test/ikatdata.target') as f:
    labels = [line.strip() for line in f]

# Use the generated hypotheses as predictions
with open('./D3test/ikatdataparacomet.hypo') as f:
    predictions = [line.strip() for line in f]

# Now you can calculate accuracy
from sklearn.metrics import accuracy_score
accuracy = accuracy_score(labels, predictions)
print(f"Accuracy: {accuracy * 100:.2f}%")
