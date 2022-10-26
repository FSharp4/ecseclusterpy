# Reverse Engineering the scipy linkage methods

Linkage acceps four arguments, three of which are optional:
- `y`: A 1D condensed distance matrix or 2D array of observation vectors.
  - If a distance matrix, it must be $n^2$-sized, where $n$ is the number of observation points.


The function returns `Z`, a $(n-1) \times 4$ matrix consisting of clusters (ordered by row index); the $i$-th cluster 


The behavior of this function is supposedly similar to the MATLAB `linkage` function. We'll investigate that as well.

