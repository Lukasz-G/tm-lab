# kernel-replication

**Q.** [`booleanization-cifar`](../booleanization-cifar/) measured fixed edge kernels at
booleanization worth +0.1409 on CIFAR-10 with the machine unchanged — the largest single effect in
this repo. Every prior positive here that was tested on a second dataset broke: core-plus-tail died
on IMDb, convolutional FPTM reversed on CIFAR-10, learned messages evaporated on CIFAR-10. Three for
three. So does this one replicate, or is it a CIFAR-10 fact?

Grayscale everywhere, so colour is out of the picture and only the kernels move. `s` held constant
across both arms of each dataset, since kernel bits widen the input and `s = round(width/S)` would
otherwise change with it.

**A. It replicates — the first positive result here that survives a change of dataset.**

| dataset | no kernels | + edge kernels | delta |
|---|---|---|---|
| MNIST | 0.9553 | 0.9668 | **+0.0115** |
| Fashion-MNIST | 0.8691 | 0.8912 | **+0.0221** |
| CIFAR-10 | 0.3528 | 0.4937 | **+0.1409** |

The effect is **ordered by task difficulty**, spanning an order of magnitude from MNIST to CIFAR-10.
That is the same shape Track D's failures had, and it reads the same way: fixed edge kernels supply
structure that raw pixel thresholding does not, and the harder the task, the less the raw
thresholding was managing on its own. Direction is a property of the encoder; magnitude is a property
of the dataset, so quote the range rather than the CIFAR-10 number alone.

**Worth putting beside the algorithm work.** Convolutional FPTM gained +0.0216 on MNIST and lost
0.0429 on CIFAR-10. Edge kernels gain +0.0115 on MNIST — *less* on that dataset — and keep gaining as
the task gets harder. The smaller MNIST number is the more trustworthy one.

**Caveats.** Three image datasets, all small and all classification. One kernel bank (Sobel x, Sobel
y, 4-neighbour Laplacian), one pooling factor, 2 thresholds on the responses, 40 clauses per class,
3 seeds. Nothing here says which kernels matter or whether a bank chosen per dataset would do better;
the claim is only that a fixed generic bank helps everywhere it was tried.
