Introduction
======================
Biased Random Forest is a method to deal with problems with unbalanced labels. The algorithm
is described in `Bader-El-Den at. al. <https://ieeexplore.ieee.org/document/8541100>`_.

Here is the python implementation of the biased random forest. I used 
`this <https://machinelearningmastery.com/implement-random-forest-scratch-python/>`_
by Jason Brownlee to build the base random forest.

Biased Random Forest
======================

Summary
---------
Python implementation of Biased Random Forest for dealing class imbalance in binary classification problems.

API
------
.. automodule:: barf.biased_rf
   :members:


Utilities
===============

Summary
--------
Utilities like making k-fold validaton tests on an estimator.

API
----
.. automodule:: barf.base
   :members: