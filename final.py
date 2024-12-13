import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (Conv2D, MaxPooling2D,
                                     Flatten, Dense, Dropout)
from tensorflow.keras.utils import to_categorical

D_path = "C:/Users/HBG/codes/project/dataset/D/data/mfcc.npy"
N_path = "C:/Users/HBG/codes/project/dataset/N/data/mfcc.npy"

mfcc_D = np.load(D_path, allow_pickle=True)
mfcc_N = np.load(N_path, allow_pickle=True)

labels_D = np.ones(len(mfcc_D))
labels_N = np.zeros(len(mfcc_N))

X = np.concatenate([mfcc_D, mfcc_N], axis=0)
y = np.concatenate([labels_D, labels_N], axis=0)

X = np.array(X, dtype=np.float32)
y = np.array(y, dtype=np.int32)

X = (X - np.min(X)) / (np.max(X) - np.min(X))

X = X[..., np.newaxis]

print(f"X shape: {X.shape}, dtype: {X.dtype}")
print(f"y shape: {y.shape}, dtype: {y.dtype}")

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2,
                                                    random_state=42)

y_train = to_categorical(y_train, num_classes=2)
y_test = to_categorical(y_test, num_classes=2)

model = Sequential([
    Conv2D(32, (3, 3), activation='relu', padding='same',
           input_shape=X_train.shape[1:]),
    MaxPooling2D((2, 2), padding='same'),
    Dropout(0.3),

    Conv2D(64, (3, 3), activation='relu', padding='same'),
    MaxPooling2D((2, 2), padding='same'),
    Dropout(0.3),

    Conv2D(128, (3, 3), activation='relu', padding='same'),
    MaxPooling2D((2, 2), padding='same'),
    Dropout(0.4),

    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.5),
    Dense(2, activation='softmax')
])

model.summary()

model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

history = model.fit(X_train, y_train, epochs=20,
                    batch_size=32, validation_split=0.2)

test_loss, test_accuracy = model.evaluate(X_test, y_test, verbose=1)
print(f"Test loss: {test_loss:.4f}")
print(f"Test Accuracy: {test_accuracy:.2f}")

model.save("mfcc_cnn_model.h5")
