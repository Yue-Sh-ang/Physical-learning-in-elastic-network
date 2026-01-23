import numpy as np
from dataclasses import dataclass
from typing import List, Tuple
import os



class ENM:
    def __init__(
        self,
        filename: str,
        taskdir=None
    ):
        self.load_graph(filename)
        
        self.vel = np.zeros_like(self.pts)
        
        if taskdir is not None:
            self.load_task(taskdir)
    
    def load_graph(self, filename: str):

        with open(filename, "r") as f:
            lines = [ln.strip() for ln in f if ln.strip()]

        idx = 0

        # --- Read points ---
        dim = int(lines[idx]); idx += 1
        n = int(lines[idx]); idx += 1

        pts = np.zeros((n, dim), dtype=np.float64)
        for i in range(n):
            pts[i] = np.fromstring(lines[idx], sep=" ", dtype=np.float64)
            idx += 1

    # --- Read edges ---
        ne = int(lines[idx]); idx += 1

        edges: List[Tuple[int, int]] = []
        k = np.zeros(ne, dtype=np.float64)
        l0 = np.zeros(ne, dtype=np.float64)

        for e in range(ne):
            node1, node2, stiff, l0val = lines[idx].split()
            edges.append((int(node1), int(node2))) 
            k[e] = float(stiff)
            l0[e] = float(l0val)
            idx += 1

        self.dim = dim
        self.n = n
        self.pts0=pts
        self.pts = pts.copy()
        self.edges = edges
        self.k = k
        self.l0 = l0
        self.ne = len(edges)

    def load_pts(self, f64: str):
        pts_new = np.fromfile(f64, dtype=np.float64)
        pts_new = pts_new.reshape((self.n, self.dim),order='F')
        self.pts = pts_new
    
    def reset(self):
        self.pts = self.pts0.copy()
        self.vel = np.zeros_like(self.pts)

    def load_k(self, f64: str):
        k_new = np.fromfile(f64, dtype=np.float64)
        self.k = k_new
    
    def load_task(self, dir: str):
        
        input_data = [] #form (edge,st,l0)
        output_data = []
        
        # Read input.txt
        with open(os.path.join(dir, "input.txt"), "r") as f:
            n = int(f.readline().strip())
            for _ in range(n):
                parts = f.readline().split()
                edge = int(parts[0])
                st = float(parts[1])
                l0 = float(parts[2])
                input_data.append((edge-1, st, l0))#change to python index
        
        # Read output.txt
        with open(os.path.join(dir, "output.txt"), "r") as f:
            n = int(f.readline().strip())
            for _ in range(n):
                parts = f.readline().split()
                edge = int(parts[0])
                st = float(parts[1])
                l0 = float(parts[2])
                output_data.append((edge-1, st, l0))#change to python index
        
        self.input = input_data
        self.output = output_data
    
    def put_strain(self, edge: int, strain: float, k: float = 100):
        u, v = self.edges[edge]
        l0 = np.linalg.norm(self.pts0[v, :] - self.pts0[u, :])
        self.l0[edge] = l0 * (1 + strain)
        self.k[edge] = k

    #Ben's critical Temperarture
    def rigidity_matrix(self,current: bool = False):
        dim = self.dim
        assert dim == 2 or dim == 3, "Dimension must be 2 or 3"
        ne = self.ne
        R = np.zeros((ne, dim * self.n))
        pts = self.pts if current else self.pts0
        

        for e in range(ne):
            i, j = self.edges[e]
            vec_ij = pts[j] - pts[i]
            dist_ij = np.linalg.norm(vec_ij)
            if dist_ij == 0: #avoid division by zero
                continue
            unit_ij = vec_ij / dist_ij
            for d in range(dim):
                R[e, dim * i + d] = -unit_ij[d]
                R[e, dim * j + d] = unit_ij[d]
        return R
    
    def jacobian_matrix(self, current: bool = False,bounded: bool = False):
        """Calculate elastic Jacobian matrix.
        
        Args:
            current: If True, cal matrix using current pts; if False, use pts0
        """
        dim = self.dim
        assert dim == 2 or dim == 3, "Dimension must be 2 or 3"
        
        J = np.zeros((dim * self.n, dim * self.n), dtype=np.float64)
        
        pts = self.pts if current else self.pts0
        k=self.k
        if bounded:
            for e,_,_ in self.input:
                k[e]=100.0
        else:
            for e,_,_ in self.input:
                k[e]=0.0
    
        for i in range(self.ne):
            u, v = self.edges[i]
            
            # Calculate distance vector and distance
            dx = pts[v] - pts[u]
            dist2 = np.dot(dx, dx)
            
            if dist2 == 0:
                continue
            
            dist = np.sqrt(dist2)
            
            ki = k[i]
            l0i = self.l0[i]
            
            term1 = ki * (1 - l0i / dist)
            term2 = ki * l0i / (dist ** 3)
            
            bu = u * dim
            bv = v * dim
            
            for a in range(dim):
                for b in range(dim):
                    val = term1 * (1 if a == b else 0) + term2 * dx[a] * dx[b]
                    J[bu + a, bu + b] += val
                    J[bv + a, bv + b] += val
                    J[bu + a, bv + b] -= val
                    J[bv + a, bu + b] -= val
        
        return J
    
    

