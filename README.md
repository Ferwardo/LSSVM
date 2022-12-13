# Subset of regressors Least-Squares SVM

This is my python implementation of the SR-LSSVM found in the PhD thesis of dr. ing. Lode Vuegen
[[1]](https://kuleuven.limo.libis.be/discovery/fulldisplay?docid=lirias2850184&context=SearchWebhook&vid=32KUL_KUL:Lirias&search_scope=lirias_profile&tab=LIRIAS&adaptor=SearchWebhook&lang=nl)
to be used in my master thesis.

The current implementation includes the following steps found in chapter 5.4 of dr. Vuegens' thesis:

* Compute step (initialise the model with just initial observations and inital prototype vectors.)
* Normal step (train on a dataset)

The implementation itself can be found in [LSSVM.py](https://github.com/Ferwardo/LSSVM/blob/main/LSSVM.py)
