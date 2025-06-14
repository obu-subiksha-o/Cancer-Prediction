# Breast Cancer Prediction using Neural Networks

This mini project uses a simple feedforward neural network built with TensorFlow/Keras to predict breast cancer malignancy based on various diagnostic features.

## Dataset

The dataset is read from a CSV file named `cancer.csv`, and includes 31 columns. The target column is:

* `diagnosis(1=m, 0=b)`: Binary label where `1` represents malignant and `0` represents benign tumors.

The remaining columns are feature measurements like `radius_mean`, `texture_mean`, etc.

## Data Preparation

The target variable `diagnosis(1=m, 0=b)` is separated from the features.

```python
x = cancer.drop(columns = ['diagnosis(1=m, 0=b)'])
y = cancer['diagnosis(1=m, 0=b)']
```

The dataset is split into training and testing sets with an 80-20 ratio:

```python
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x , y , test_size = 0.2)
```

## Model Architecture

A sequential neural network is built using Keras:

* **Input Layer**: Dense layer with 256 units and sigmoid activation.
* **Hidden Layer**: Dense layer with 256 units and sigmoid activation.
* **Output Layer**: Dense layer with 1 unit and sigmoid activation (for binary classification).

```python
model = tf.keras.models.Sequential()
model.add(tf.keras.layers.Dense(256, input_shape = x_train.shape[1:], activation = 'sigmoid'))
model.add(tf.keras.layers.Dense(256, activation='sigmoid'))
model.add(tf.keras.layers.Dense(1, activation = 'sigmoid'))
```

## Compilation

The model is compiled using the **Adam** optimizer and **binary cross-entropy** loss function.

```python
model.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])
```

## Training

The model is trained for **700 epochs** on the training data:

```python
model.fit(x_train, y_train, epochs = 700)
```

## Evaluation

The model is evaluated on the test set:

```python
model.evaluate(x_test, y_test)
```

**Test Accuracy:** \~93.86%
**Test Loss:** \~0.2081
