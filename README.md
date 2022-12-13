# Subset of regressors Least-Squares SVM

This is my python implementation of the SR-LSSVM found in the PhD thesis of dr. ing. Lode Vuegen
[[1]](https://kuleuven.limo.libis.be/discovery/fulldisplay?docid=lirias2850184&context=SearchWebhook&vid=32KUL_KUL:Lirias&search_scope=lirias_profile&tab=LIRIAS&adaptor=SearchWebhook&lang=nl)
to be used in my master thesis.

The current implementation includes the following steps found in chapter 5.4 of dr. Vuegens' thesis:

* Compute step:

```python
compute(X_init, Y_init, X_pv=None, Y_pv=None, C=None)

# Compute the model parameters from scratch
# 
# X_init: The observations used to initialise the model
# Y_init: The labels for the corresponding observations of X_init
# X_pv: The prototype vectors used to initialise the model, optional
# Y_init: The labels for the corresponding observations of X_pv, optional
# C: Regularisation parameter, optional
```

* Normal step:

```python
normal(X, Y)

# Does a normal training step on the model with the given dataset
# 
# X: The observations to train on
# Y: The corresponding class labels of X
```

There is also a function to predict a label:

```python
predict(x)

# Predicts the label of the given observation
# 
# x: A single vector with an observation to be classified.
# returns: The predicted class label
```

The implementation itself can be found in [LSSVM.py](https://github.com/Ferwardo/LSSVM/blob/main/LSSVM.py)
