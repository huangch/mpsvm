# MPSVM: Massively Parallelized Support Vector Machines based on GPU-Accelerated Multiplicative Updates

we present multiple parallelized support vector machines (MPSVMs), which aims to deal with the situation when multiple SVMs are required to be performed concurrently. The proposed MPSVM is based on an optimiza- tion procedure for nonnegative quadratic programming (NQP), called multiplicative updates. By using graphical processing units (GPUs) to parallelize the numerical procedure of SVMs, the proposed MPSVM showed good performance for a certain range of data size and dimension.

In the experiments, we compared the proposed MPSVM with other cutting-edge implementations of GPU-based SVMs and it showed competitive performance. Furthermore, the proposed MPSVM is designed to perform multiple SVMs in parallel. As a result, when multiple operations of SVM are required, MPSVM can be one of the best options in terms of time consumption.
