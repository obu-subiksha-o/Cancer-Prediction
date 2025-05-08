# Cancer-Prediction

A TensorFlow-based binary classifier to predict breast cancer diagnosis using the Wisconsin Breast Cancer Dataset. The model distinguishes between **malignant** and **benign** tumors with high accuracy.

## Dataset

* **Source:** [Breast Cancer Wisconsin (Diagnostic) Data Set](https://archive.ics.uci.edu/ml/datasets/Breast+Cancer+Wisconsin+%28Diagnostic%29)
* **Target variable:**

  * `diagnosis(1=m, 0=b)` — 1 = Malignant, 0 = Benign
* **Features:** 30 numerical features computed from digitized images of fine needle aspirate (FNA) of breast mass.

## Features

* Cleaned and preprocessed data using **Pandas**
* Neural Network classifier with **TensorFlow / Keras**
* Achieved **93.86%** accuracy on test data
* End-to-end implementation on **Google Colab**

## 🛠️ Requirements

* Python 3.x
* TensorFlow
* Scikit-learn
* Pandas

Install dependencies:

```bash
pip install tensorflow scikit-learn pandas
```

## 📝 Project Workflow

1. **Load and inspect data**

```python
import pandas as pd

cancer = pd.read_csv('cancer.csv')
cancer.head()
```

2. **Split features and labels**

```python
x = cancer.drop(columns=['diagnosis(1=m, 0=b)'])
y = cancer['diagnosis(1=m, 0=b)']
```

3. **Train-test split**

```python
from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)
```

4. **Build the model**

```python
import tensorflow as tf

model = tf.keras.models.Sequential([
    tf.keras.layers.Dense(256, input_shape=x_train.shape[1:], activation='sigmoid'),
    tf.keras.layers.Dense(256, activation='sigmoid'),
    tf.keras.layers.Dense(1, activation='sigmoid')
])
```

5. **Compile the model**

```python
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
```

6. **Train the model**

```python
model.fit(x_train, y_train, epochs=700)
```

7. **Evaluate**

```python
model.evaluate(x_test, y_test)
```

Expected output:

```
Test Accuracy ≈ 93.86%
```

## Model Architecture

| Layer          | Neurons | Activation |
| -------------- | ------- | ---------- |
| Dense (Input)  | 256     | Sigmoid    |
| Dense (Hidden) | 256     | Sigmoid    |
| Dense (Output) | 1       | Sigmoid    |

## Results

| Metric            | Score   |
| ----------------- | ------- |
| **Test Accuracy** | 93.86%  |
| **Loss**          | \~0.208 |

## Notes

* The model uses **sigmoid activation** in all layers — effective but experimenting with **ReLU** or other activations could improve results.
* **700 epochs** were used — early stopping or tuning could optimize training time.

