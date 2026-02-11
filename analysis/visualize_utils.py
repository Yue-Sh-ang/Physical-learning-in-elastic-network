import numpy as np
from dataclasses import dataclass
from typing import List, Tuple
import os
import matplotlib.pyplot as plt
from scipy.linalg import orthogonal_procrustes as procrustes


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
        k=self.k.copy()
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
    
    def rigid_correction(self,pts):
        b = self.pts0 - np.mean(self.pts0, axis=0)
        a = pts - np.mean(pts, axis=0)
        R, sca = procrustes(a, b, check_finite=False)
        pts_new= a @ R
        return pts_new

    # def mode_projection(self, modes: np.ndarray):
    #     """Project current displacement onto given modes.
    #     Args:
    #         modes: Array of shape (n_modes, dim * n)
        
    #     Returns:
    #         coeffs: Array of shape (n_modes,)
    #     """
    #     self.rigid_correction()
    #     disp = self.pts.ravel() - self.pts0.ravel()
    #     coeffs = modes @ disp
    #     return coeffs

    def plot_2d(self,ax=None,color=None,vmin=None,vmax=None,current: bool = True):
        if self.dim != 2:
            raise ValueError("Dimension must be 2 for 2D plotting.")
        if current:
            pts = self.pts
        else:
            pts = self.pts0

        if color is None:
            color='gray'
        else:
            assert color.size==(self.ne)
            color=np.array(color)
            if vmin is None:
                vmin=color.min()
            if vmax is None:
                vmax=color.max()
            color=plt.cm.bwr((color-vmin)/(vmax-vmin))
                

        if ax is None:
            fig, ax = plt.subplots()
        #plot the nodes color if there is special nodes(input nodes or output nodes)
        skip_edge=[]
        ax.scatter(pts[:, 0], pts[:, 1], c='black', s=20,zorder=1)
        if self.input is not None:
            for edge, _, _ in self.input:
                skip_edge.append(edge)
                u, v = self.edges[edge]
                ax.scatter(pts[u, 0], pts[u, 1], c='blue', s=30,zorder=2)
                ax.scatter(pts[v, 0], pts[v, 1], c='blue', s=30,zorder=2)
        if self.output is not None:
            for edge, _, _ in self.output:
                skip_edge.append(edge)
                u, v = self.edges[edge]
                ax.scatter(pts[u, 0], pts[u, 1], c='red', s=30,zorder=2)
                ax.scatter(pts[v, 0], pts[v, 1], c='red', s=30,zorder=2)
        
        for i, (u, v) in enumerate(self.edges):
            if i in skip_edge:
                continue
            x = [pts[u, 0], pts[v, 0]]
            y = [pts[u, 1], pts[v, 1]]
            ax.plot(x, y, c=color, linewidth=1,zorder=5)
        ax.set_aspect('equal')

    def plot_disp(self,disp,ax=None,scale=1.0, rigid: bool = False):
        dim=self.dim
        pts0 = self.pts0
        pts=pts0+scale*disp.reshape((self.n,self.dim))
        if rigid:
            pts = self.rigid_correction(pts)
        if ax is None:
            fig, ax = plt.subplots()
        
        ax.scatter(pts[:, 0], pts[:, 1], c='black', s=20,zorder=1)
        ax.scatter(pts0[:, 0], pts0[:, 1], c='grey', s=20,zorder=1)

        skip_edge=[]
        if self.input is not None:
            for edge, _, _ in self.input:
                skip_edge.append(edge)
                u, v = self.edges[edge]
                ax.scatter(pts[u, 0], pts[u, 1], c='blue', s=30,zorder=2)
                ax.scatter(pts[v, 0], pts[v, 1], c='blue', s=30,zorder=2)
                ax.scatter(pts0[u, 0], pts0[u, 1], c='blue', s=30,zorder=2)
                ax.scatter(pts0[v, 0], pts0[v, 1], c='blue', s=30,zorder=2)
        if self.output is not None:
            for edge, _, _ in self.output:
                skip_edge.append(edge)
                u, v = self.edges[edge]
                ax.scatter(pts[u, 0], pts[u, 1], c='red', s=30,zorder=2)
                ax.scatter(pts[v, 0], pts[v, 1], c='red', s=30,zorder=2)
                ax.scatter(pts0[u, 0], pts0[u, 1], c='red', s=30,zorder=2)
                ax.scatter(pts0[v, 0], pts0[v, 1], c='red', s=30,zorder=2)
        
        for i, (u, v) in enumerate(self.edges):
            if i in skip_edge:
                continue
            x = [pts[u, 0], pts[v, 0]]
            y = [pts[u, 1], pts[v, 1]]
            ax.plot(x, y, c='black', linewidth=1,zorder=5)  

            x0 = [pts0[u, 0], pts0[v, 0]]
            y0 = [pts0[u, 1], pts0[v, 1]]
            ax.plot(x0, y0, c='gray', linestyle='-', linewidth=1,zorder=4)
        ax.set_aspect('equal')
    




