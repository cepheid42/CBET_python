from mpi4py import MPI
import numpy as np

comm = MPI.COMM_WORLD
size = comm.Get_size()
rank = comm.Get_rank()

a = np.zeros((20, 20), dtype=np.int)

for i in range(rank, 20, size):
    # a[i, :] += rank
    if rank != 0:
        comm.send(i, dest=0, tag=10)
        temp1 = np.zeros(20, dtype=np.int)
        temp1[:] += rank
        comm.Send(temp1, dest=0, tag=11)
    else:
        for r in range(1, size):
            index = comm.recv(source=r, tag=10)
            # print(f'{r} - {index}')
            temp = np.empty(20, dtype=np.int)
            comm.Recv(temp, source=r, tag=11)
            a[index, :] = temp
if rank == 0:
    print(a)






