# Review of Graph Spectrum Theory

_Mar 2019_

### The Graph Laplacian Matrix
- Incidence matrix C=C(G), each row is an edge, and each column is a vertex, the source is +1, and sink is -1. 
- Graph Laplacian matrix $L(G) = C^T C = D-W$. D diagonal matrix records the degrees and W is the adjacency matrix.
- L(G) is symmetric, thus L(G) has real-valued, non-negative eigenvalues, and real-valued orthogonal eigenvalues.
- G has K connected components iff there are k eigenvalues that are 0.
- Partition vector: Let x represent a bipartition of a graph. Each element is a vertex, whether it is in one partition (+1) or the other (-1). 
- The number of cut edges (edges with vertices in two partitions) are $\frac{1}{4} x^T L(G) x$.
- If we want to minimize the cut between partitions, then use $q_1$ as the partition vector, the # cut is then $\frac{1}{4} n \lambda_1$. The partition vector is then $\texttt{sign}(q_1)$
- [Source](https://www.youtube.com/watch?v=rVnOANM0oJE)
