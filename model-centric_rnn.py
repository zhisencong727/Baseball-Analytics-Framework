from tf_keras.models import Sequential
from tf_keras.layers import Masking, LSTM, Dense
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from pitch_seq_data_preq import get_data

def get_sequence_lengths(sequence, padding_value=0.0):
    """
    Calculates the actual lengths of sequences before padding.
    """
    for i in range(len(sequence)):
        if sum(sequence[i]) == 0.0:
            return i + 1
    return len(sequence)

def random_crop_and_pad_padded_sequence(sequence, original_length, desired_length):
    crop_length = np.random.randint(2, original_length)
    max_start = original_length - crop_length - 1
    start_idx = np.random.randint(0, max_start + 1)
    end_idx = start_idx + crop_length
    cropped_sequence = sequence[start_idx:end_idx]
    output = -1.0
    for i in range(len(cropped_sequence) - 1, 0, -1):
        if sum(cropped_sequence[i]) != 0.0:
            output = cropped_sequence[i][0]
            cropped_sequence[i] = [0.0] * 10
    if output == -1.0:
        print("CROPPED SEQUENCE:")
        print(cropped_sequence)
        print("CROPPED LENGTH")
        print(crop_length)
    while len(cropped_sequence) < desired_length:
        padding = [0.0] * 10
        cropped_sequence.append(padding)
    return [cropped_sequence, output]

# Load data
training_data, max_length = get_data(False)
input_sequences = []
targets = []
augmentation = False

for at_bat in training_data:
    sequence, target = at_bat
    input_sequences.append(sequence)
    targets.append(target)
    if augmentation:
        sequence_length = get_sequence_lengths(sequence)
        if sequence_length > 2:
            augmented_sequence, augmented_output = random_crop_and_pad_padded_sequence(
                sequence, sequence_length, max_length
            )
            input_sequences.append(augmented_sequence)
            targets.append(augmented_output)
            
input_sequences = np.array(input_sequences, dtype='float32')
targets = np.array(targets, dtype='float32')

# Encode targets
from sklearn.preprocessing import LabelEncoder

label_encoder = LabelEncoder()
targets_encoded = label_encoder.fit_transform(targets)

# Split data BEFORE applying sampling
X_train, X_val, y_train, y_val = train_test_split(
    input_sequences, targets_encoded, test_size=0.25, random_state=42, stratify=targets_encoded
)

# Apply Oversampling and/or Undersampling on the training data
from imblearn.over_sampling import RandomOverSampler
from imblearn.under_sampling import RandomUnderSampler
from imblearn.pipeline import Pipeline

# Define the sampling strategy
# Example: First oversample minority classes, then undersample majority classes
over = RandomOverSampler(sampling_strategy='auto', random_state=42)
under = RandomUnderSampler(sampling_strategy='auto', random_state=42)

steps = [('over', over)]
pipeline = Pipeline(steps=steps)

# Reshape X_train to 2D for sampling
num_train_samples, time_steps, num_features = X_train.shape
X_train_flat = X_train.reshape(num_train_samples, -1)

# Apply the pipeline to X_train and y_train
X_train_resampled, y_train_resampled = pipeline.fit_resample(X_train_flat, y_train)

# Reshape X_train_resampled back to 3D
X_train_resampled = X_train_resampled.reshape(-1, time_steps, num_features)

# Optionally, verify the new class distribution
from collections import Counter
print("Resampled training set class distribution:", Counter(y_train_resampled))

# Encode targets to one-hot after resampling
from tf_keras.utils import to_categorical
num_classes = len(np.unique(y_train_resampled))
y_train_resampled_one_hot = to_categorical(y_train_resampled, num_classes=num_classes)

# Encode y_val to one-hot
y_val_one_hot = to_categorical(y_val, num_classes=num_classes)

# Apply Standard Scaling
from sklearn.preprocessing import StandardScaler

# Flatten the training data for scaling
X_train_resampled_flat = X_train_resampled.reshape(-1, num_features)

# Fit scaler on resampled training data
scaler = StandardScaler()
X_train_scaled_flat = scaler.fit_transform(X_train_resampled_flat)

# Reshape back to 3D
X_train_scaled = X_train_scaled_flat.reshape(-1, time_steps, num_features)

# Transform validation data
num_val_samples = X_val.shape[0]
X_val_flat = X_val.reshape(-1, num_features)
X_val_scaled_flat = scaler.transform(X_val_flat)
X_val_scaled = X_val_scaled_flat.reshape(num_val_samples, time_steps, num_features)

y_train_resampled_one_hot = y_train_resampled_one_hot.astype('float32')
y_val_one_hot = y_val_one_hot.astype('float32')

# Build the model
from tf_keras.regularizers import l2
from tf_keras.layers import Dropout

model = Sequential([
    Masking(mask_value=0.0, input_shape=(time_steps, num_features)),
    LSTM(1024, activation='relu', kernel_regularizer=l2(0.01), return_sequences=True),
    Dropout(0.1),
    LSTM(512, activation='relu'),
    Dropout(0.1),
    Dense(num_classes, activation='softmax')
])

from tf_keras.optimizers import RMSprop

optimizer = RMSprop(learning_rate=0.005)

# Compile the model
model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])

from tf_keras.callbacks import ReduceLROnPlateau, EarlyStopping, ModelCheckpoint

reduce_lr = ReduceLROnPlateau(
    monitor='val_loss', factor=0.5, patience=5, min_lr=1e-6, verbose=1
)
early_stopping = EarlyStopping(
    monitor='val_loss', patience=10, restore_best_weights=True, verbose=1
)
checkpoint = ModelCheckpoint(
    filepath='best_model.h5',
    monitor='val_accuracy',
    verbose=1,
    save_best_only=True,
    mode='max'
)
callbacks = [checkpoint]

# Train the model
history = model.fit(
    X_train_scaled, y_train_resampled_one_hot,
    epochs=75,
    batch_size=32,
    validation_data=(X_val_scaled, y_val_one_hot),
    callbacks=callbacks
    # You can also include `reduce_lr` and `early_stopping` in callbacks if desired
    # callbacks=[checkpoint, reduce_lr, early_stopping]
)

from tf_keras.models import load_model
best_model = load_model('best_model.h5')

# Evaluate the model
val_loss, val_acc = best_model.evaluate(X_val_scaled, y_val_one_hot)
print(f"Validation Loss: {val_loss:.4f}, Validation Accuracy: {val_acc:.4f}")

# Predictions
y_pred_probs = best_model.predict(X_val_scaled)
y_pred_classes = np.argmax(y_pred_probs, axis=1)
y_pred_labels = label_encoder.inverse_transform(y_pred_classes)
unique_targets, counts = np.unique(y_pred_labels, return_counts=True)
print("Counts of predicted target values:")
for target, count in zip(unique_targets, counts):
    print(f"Class {target}: {count}")

y_val_classes = np.argmax(y_val_one_hot, axis=1)
y_val_original_labels = label_encoder.inverse_transform(y_val_classes)
unique_labels, counts = np.unique(y_val_original_labels, return_counts=True)
class_counts = dict(zip(unique_labels, counts))

# Print the class counts with original labels
print("Class counts in y_val:")
for label, count in class_counts.items():
    print(f"{label}: {count} samples")

# Classification report
from sklearn.metrics import classification_report

print("\nClassification Report (Using Original Class Names):")
print(classification_report(y_val_original_labels, y_pred_labels))

# Save the trained model
best_model.save('rnn_model.h5')

from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

cm = confusion_matrix(y_val_original_labels, y_pred_labels, labels=label_encoder.classes_)
print(cm)
cm_df = pd.DataFrame(cm, index=label_encoder.classes_, columns=label_encoder.classes_)

plt.figure(figsize=(10, 7))
sns.heatmap(cm_df, annot=True, fmt='d', cmap='Blues')
plt.ylabel('True Label')
plt.xlabel('Predicted Label')
plt.title('Confusion Matrix')
plt.show()

y_val_pred_probs = model.predict(X_val_scaled)
y_val_pred_max_probs = np.max(y_val_pred_probs, axis=1)

# Determine Threshold using 95th Percentile of ID Validation Probabilities
threshold_percentile = 65  # Adjust as needed
softmax_threshold = np.percentile(y_val_pred_max_probs, threshold_percentile)
print(f"Softmax Threshold (95th percentile): {softmax_threshold:.4f}")

# Save the threshold for future use
import pickle
with open('softmax_threshold.pkl', 'wb') as f:
    pickle.dump(softmax_threshold, f)

# Define OOD Prediction Function
def predict_with_ood(model, data, scaler, threshold):
    num_samples, time_steps, num_features = data.shape
    data_flat = data.reshape(-1, num_features)
    data_scaled_flat = scaler.transform(data_flat)
    data_scaled = data_scaled_flat.reshape(num_samples, time_steps, num_features)

    softmax_probs = model.predict(data_scaled)
    max_probs = np.max(softmax_probs, axis=1)
    preds = np.argmax(softmax_probs, axis=1)

    is_ood = max_probs < threshold
    final_preds = preds.copy()
    final_preds[is_ood] = -1  # -1 indicates OOD

    return final_preds, is_ood

# Example Usage on Validation Set
final_predictions, ood_flags = predict_with_ood(model, X_val, scaler, softmax_threshold)
print(f"Number of OOD samples in Validation Set: {np.sum(ood_flags)} out of {len(ood_flags)}")

# Save the scaler for future use
with open('scaler.pkl', 'wb') as f:
    pickle.dump(scaler, f)

# Save the label encoder
with open('label_encoder.pkl', 'wb') as f:
    pickle.dump(label_encoder, f)

# -------------------- OOD Detection Integration Ends Here ---------------------

# ... [Rest of your existing code, such as evaluation and visualization]

import matplotlib.pyplot as plt
import seaborn as sns

sns.histplot(y_val_pred_max_probs, bins=50, kde=True)
plt.axvline(softmax_threshold, color='red', linestyle='--', label='Threshold')
plt.title('Distribution of Maximum Softmax Probabilities')
plt.xlabel('Max Softmax Probability')
plt.ylabel('Frequency')
plt.legend()
plt.show()



in_distribution_mask = ~ood_flags  # True for ID samples
X_val_id = X_val_scaled[in_distribution_mask]
y_val_id = y_val_original_labels[in_distribution_mask]
y_pred_id = y_pred_labels[in_distribution_mask]

# Generate Classification Report for In-Distribution Data
print("\nClassification Report for In-Distribution Data:")
print(classification_report(y_val_id, y_pred_id))

# Optionally, you can also display the number of ID samples
print(f"Number of In-Distribution samples: {len(y_val_id)} out of {len(y_val)}")
