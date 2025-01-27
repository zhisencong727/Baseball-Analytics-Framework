from pitch_seq_data_preq import get_data
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    top_k_accuracy_score
)
import matplotlib.pyplot as plt
import seaborn as sns

# Set random seed for reproducibility
np.random.seed(42)

# Load data
training_data, max_length = get_data(False)
input_sequences = []
targets = []
sequence_len_set = set()
for at_bat in training_data:
    sequence, target = at_bat
    temp = []
    for each in sequence:
        for e in each:
            temp.append(e)
    print(temp)
    input_sequences.append(temp)
    targets.append(target)        
X = np.array(input_sequences, dtype='float32')
y = np.array(targets, dtype='float32')
print(X.shape)
print(y.shape)
print(sequence_len_set)

unique, counts = np.unique(y, return_counts=True)
distribution = dict(zip(unique, counts))
print(distribution)

# Encode labels if they are categorical
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)



# Split the data into training and testing sets with stratification
X_train, X_test, y_train, y_test = train_test_split(
    X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

"""
from imblearn.over_sampling import SMOTE
# Apply SMOTE to training data
smote = SMOTE(random_state=42)
X_train_resampled, y_train_resampled = smote.fit_resample(X_train_scaled, y_train)


from imblearn.under_sampling import RandomUnderSampler

rus = RandomUnderSampler(random_state=42)
X_resampled, y_resampled = rus.fit_resample(X_train, y_train)

from sklearn.feature_selection import SelectKBest, f_classif

# Initialize SelectKBest with ANOVA F-test
selector = SelectKBest(score_func=f_classif, k=10)  # 'k' can be tuned

# Fit to training data and transform
X_train_selected = selector.fit_transform(X_train_resampled, y_train_resampled)
X_test_selected = selector.transform(X_test_scaled)
"""


mlp = MLPClassifier(
    hidden_layer_sizes=(100,50),          # Single hidden layer with 100 neurons
    activation='relu',                  # Activation function for hidden layers
    solver='adam',                      # Optimizer
    alpha=0.0001,                       # L2 penalty (regularization term)
    batch_size='auto',                  # Size of minibatches for stochastic optimizers
    learning_rate='constant',           # Learning rate schedule for weight updates
    learning_rate_init=0.001,           # Initial learning rate
    max_iter=300,                       # Maximum number of iterations
    random_state=42,                    # Seed for random number generator
    early_stopping=True,                # Whether to use early stopping to terminate training
    validation_fraction=0.1,            # Proportion of training data to set aside as validation
    n_iter_no_change=50,                # Number of epochs with no improvement to wait before stopping
    
)




# Train the model
mlp.fit(X_train, y_train)
X_test_scaled = X_test
# Predict on the test set
y_pred = mlp.predict(X_test_scaled)
y_pred_decoded = label_encoder.inverse_transform(y_pred)
unique, counts = np.unique(y_pred_decoded, return_counts=True)
distribution = dict(zip(unique, counts))
print(distribution)

# Predict probabilities
y_proba = mlp.predict_proba(X_test_scaled)
#print(y_proba*100)
# Calculate standard accuracy
accuracy = accuracy_score(y_test, y_pred)
from sklearn.metrics import f1_score
f1 = f1_score(y_test,y_pred,average='weighted')
print("F1 Score is:",f1)

print(f"\nTest Accuracy (Deterministic): {accuracy * 100:.2f}%\n")

# Calculate Top-2 Accuracy
top2_accuracy = top_k_accuracy_score(y_test, y_proba, k=2)
print(f"Top-2 Accuracy: {top2_accuracy * 100:.2f}%\n")

# Confusion Matrix for deterministic predictions
print("Confusion Matrix (Deterministic):")
y_test_decoded = label_encoder.inverse_transform(y_test)
cm = confusion_matrix(y_test_decoded, y_pred_decoded)
print(cm)

# Visualization of the Confusion Matrix
plt.figure(figsize=(10, 7))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=label_encoder.classes_,
            yticklabels=label_encoder.classes_)
plt.xlabel('Predicted')
plt.ylabel('True')
plt.title('Confusion Matrix (Deterministic Predictions)')
plt.show()

# Function to compute Top-2 Confusion Matrix (Optional)
# Note: Standard confusion matrices don't directly support Top-k.
# This requires custom implementation if needed.

# Optional: Top-2 Prediction Analysis
# Identify instances where the true label is in the top 2 predictions
top2_preds = np.argsort(y_proba, axis=1)[:, -2:]
correct_top2 = [y_test[i] in top2_preds[i] for i in range(len(y_test))]
top2_correct = sum(correct_top2)
print(f"Top-2 Correct Predictions: {top2_correct} out of {len(y_test)}")
print(f"Top-2 Accuracy (Manual Calculation): {top2_correct / len(y_test) * 100:.2f}%")

# Optional: Detailed Analysis of Top-2 Predictions
# For example, how often the true label is the second choice
second_choice_correct = [
    y_test[i] == top2_preds[i][0] or y_test[i] == top2_preds[i][1]
    for i in range(len(y_test))
]
print(f"Top-2 Accuracy (Alternative Calculation): {sum(second_choice_correct) / len(y_test) * 100:.2f}%")