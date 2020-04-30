# Parallel Computation of Cross Beam Energy Transfer

## Requirements
- Python 3.7+
- numpy 1.16+
- mpi4py 3.0.3
- Yorick 2.2.04x

## Instructions
There are four versions of this code in four folders.
The Yorick folder contains the original code given by Dr. Adam Sefkow, with modifications for testing.

The other three versions are Sequential, MPI, and Pool. These represent the unparallelized version, 
the version that uses mpi4py, and the version that uses the Python Multiprocessing library.

To run the sequential and pool versions, the command is
```
python3 seq_cbet.py
```
and
```
python3 pool_cbet.py
```
The number of processes in pool can be modified by changing the `num_procs` variable at the top of the pool_cbet.py file.


To run the MPI version, the command is

```
mpiexec -n 4 python3 mpi_cbet.py
```
where the number following the -n is the number of processes to use.
