import numpy as np
"""
==============================================================================
==============================================================================
==============================================================================

Implementation and Comparative Analysis of Soft-Margin SVM using Primal and Dual Optimization Techniques

==============================================================================
==============================================================================
==============================================================================
SOFT-MARGIN SUPPORT VECTOR MACHINE (SVM) FROM SCRATCH
(PRIMAL & DUAL OPTIMIZATION FRAMEWORKS)

This project implements a complete Soft-Margin SVM from scratch without using
machine learning libraries such as scikit-learn. The implementation explores
both the primal and dual formulations of SVM optimization and provides a
comparative study of different gradient-based optimization techniques.

Implemented Features
--------------------
1. Primal Optimization
   - Batch Gradient Descent (BGD)
   - Stochastic Gradient Descent (SGD)
   - Mini-Batch Gradient Descent (Mini-BGD)

2. Dual Optimization
   - Batch Gradient Descent (BGD)
   - Stochastic Gradient Descent (SGD)
   - Mini-Batch Gradient Descent (Mini-BGD)

3. Core Soft-Margin SVM Components
   - Hinge Loss Minimization
   - L2 Regularization
   - Slack Variable Computation
   - Margin Distance Evaluation
   - Support Vector Identification
   - Free Support Vectors (0 < α < C)
   - Boundary Support Vectors (α = C)

4. Model Analysis and Evaluation
   - Training Accuracy
   - Duality Gap Analysis
   - Primal-Dual Solution Comparison
   - Support Vector Analysis
   - Decision Boundary Visualization

Mathematical Formulation
------------------------
Primal Objective:

    min (1/2)||w||² + C Σ ξ_i

subject to

    y_i(wᵀx_i + b) ≥ 1 − ξ_i
    ξ_i ≥ 0

Dual Objective:

    max Σ α_i − (1/2)αᵀGα

subject to

    0 ≤ α_i ≤ C
    Σ α_i y_i = 0

where

    G_ij = y_i y_j (x_iᵀx_j)

Highlights
----------
• Built entirely from scratch using NumPy.
• Implemented and compared BGD, SGD, and Mini-BGD in both primal and dual forms.
• Reconstructed primal parameters (w, b) from optimized dual variables.
• Evaluated optimization quality using duality-gap analysis.
• Compared primal and dual solutions on overlapping/non-separable datasets.
• Visualized support-vector behavior and decision boundaries.

Purpose
-------
To study Soft-Margin SVMs, Hinge Loss, KKT Conditions, Lagrangian Duality,
Convex Optimization, and Gradient-Based Optimization methods through a fully
custom implementation.

===============================================================================
"""
class SoftMargin_dual_primal_system:
    def __init__(self,x,y,c=1.0):
        self.x=x
        if self.x.ndim==1:
            self.x=self.x.reshape(-1,1)
        self.m,self.n=self.x.shape
        self.y=y.reshape(-1,1)
        if self.y.shape[0]!=self.m:
            raise ValueError("x and y must have same number of samples")
        if not np.all(np.isin(self.y,[-1,1])):
            raise ValueError("y must contain only -1 and +1")
        if np.unique(self.y).size!=2:
            raise ValueError("Both classes must be present")
        self.c=c

        self.alpha=None
        self.beta=None

        self.w=None
        self.b=None

        self.slack_i=None
        self.u_i=None

        x_signed=self.y*self.x
        self.G=x_signed@x_signed.T

        self.P=None
        self.D=None

    def margin_distance_score(self):
        self.u_i= self.y*(self.x@self.w+self.b)
        return self.u_i
    
    def primal_bgd_method(self,lr=0.01,max_iter=1000):
        self.w=np.zeros((self.n,1))
        self.b=0
        margin_distance=self.margin_distance_score()
        lamb=1/self.c/self.m
        for _ in range(max_iter):
            mask=(margin_distance<1).ravel()
            w_grad=lamb*self.w-1/self.m*(np.sum(self.y[mask]*self.x[mask],axis=0)).reshape(-1,1)
            b_grad=-1/self.m*np.sum(self.y[mask],axis=0)
            if np.linalg.norm(w_grad)<1e-6 and np.linalg.norm(b_grad)<1e-6:
                print(f"converged early at iteration {_}")
                break
            self.w-=lr*w_grad
            self.b-=lr*b_grad
            margin_distance=self.margin_distance_score()

    def dual_bgd_method(self,lr=0.01,max_iter=1000):
        self.alpha=np.zeros((self.m,1))
        self.beta=0.0
        for _ in range(max_iter):
            alpha_grad=np.ones((self.m,1))-self.G@self.alpha-self.beta*self.y
            beta_grad=-(self.y.T@self.alpha).item()
            if np.linalg.norm(alpha_grad)<1e-6 and np.linalg.norm(beta_grad)<1e-6:
                print(f"converged early at iteration {_}")
                break
            self.alpha+=lr*alpha_grad
            self.alpha=np.clip(self.alpha,0,self.c)
            self.beta-=lr*beta_grad

    def primal_sgd_method(self,lr=0.01,max_iter=1000):
        self.w=np.zeros((self.n,1))
        self.b=0
        lamb=1/self.c/self.m
        margin_distance=self.margin_distance_score()
        for _ in range(max_iter):
            idx=np.random.randint(0,self.m)
            if margin_distance[idx,0]<1:
                w_grad=lamb*self.w-1*(self.y[idx]*self.x[idx]).reshape(-1,1)
                b_grad=-1*self.y[idx,0]
            else:
                w_grad=lamb*self.w
                b_grad=0
            self.w-=lr*w_grad
            self.b-=lr*b_grad
            margin_distance=self.margin_distance_score()

    def dual_sgd_method(self,lr=0.01,max_iter=1000):
        self.alpha=np.zeros((self.m,1))
        self.beta=0.0
        for _ in range(max_iter):
            i=np.random.randint(0,self.m)
            alpha_grad=(1-self.G[i:i+1,:]@self.alpha-self.beta*self.y[i,0]).item()
            self.alpha[i,0]+=lr*alpha_grad
            self.alpha[i,0]=np.clip(self.alpha[i,0],0,self.c)
            beta_grad=(self.y.T@self.alpha).item()
            self.beta+=lr*beta_grad


    def primal_mini_bgd_method(self,lr=0.01,max_iter=1000,batch_size=32):
        self.w=np.zeros((self.n,1))
        self.b=0
        lamb=1/self.c/self.m
        margin_distance=self.margin_distance_score()
        for _ in range(max_iter):
            idx=np.random.permutation(self.m)
            for i in range(0,self.m,batch_size):
                batch_idx=idx[i:i+batch_size]
                current_batch_size=len(batch_idx)
                if current_batch_size==0:
                    continue
                mask=(margin_distance[batch_idx]<1).ravel()
                if np.any(mask):
                    w_grad=lamb*self.w-1/current_batch_size*(np.sum(self.y[batch_idx][mask]*self.x[batch_idx][mask],axis=0)).reshape(-1,1)
                    b_grad=-1/current_batch_size*np.sum(self.y[batch_idx][mask],axis=0)
                else:
                    w_grad=lamb*self.w
                    b_grad=0
                self.w-=lr*w_grad
                self.b-=lr*b_grad
                margin_distance=self.margin_distance_score()
    
    def dual_mini_bgd_method(self,lr=0.01,max_iter=1000,batch_size=32):
        self.alpha=np.zeros((self.m,1))
        self.beta=0.0
        for _ in range(max_iter):
            idx=np.random.permutation(self.m)
            for i in range(0,self.m,batch_size):
                batch_idx=idx[i:i+batch_size]
                current_batch_size=len(batch_idx)
                if current_batch_size==0:
                    continue
                alpha_grad=np.ones((current_batch_size,1))-self.G[batch_idx,:]@self.alpha-self.beta*self.y[batch_idx]
                self.alpha[batch_idx]+=lr*alpha_grad
                self.alpha[batch_idx]=np.clip(self.alpha[batch_idx],0,self.c)
                beta_grad=-(self.y.T@self.alpha).item()
                self.beta-=lr*beta_grad
        
    def primal_solver(self,lr=0.01,max_iter=1000,batch_size=32,method='bgd'):
        if method=='bgd':
            self.primal_bgd_method(lr,max_iter)
        elif method=='sgd':
            self.primal_sgd_method(lr,max_iter)
        elif method=='mini-bgd':
            self.primal_mini_bgd_method(lr,max_iter,batch_size)
        else:
            raise ValueError("method must be 'bgd', 'sgd' or 'mini-bgd'")
        
    def dual_solver(self,lr=0.01,max_iter=1000,batch_size=32,method='bgd'):
        if method=='bgd':
            self.dual_bgd_method(lr,max_iter)
        elif method=='sgd':
            self.dual_sgd_method(lr,max_iter)
        elif method=='mini-bgd':
            self.dual_mini_bgd_method(lr,max_iter,batch_size)
        else:
            raise ValueError("method must be 'bgd', 'sgd' or 'mini-bgd'")
        self.w=self.x.T@(self.alpha*self.y)
        free_sv=self.free_support_vectors()
        if len(free_sv)>0:
            self.b=np.mean(self.y[free_sv,0]-(self.x[free_sv]@self.w).ravel())
        else:
            boundary_sv=self.boundary_support_vectors()
            if len(boundary_sv)>0:
                self.b=np.mean(self.y[boundary_sv,0]-(self.x[boundary_sv]@self.w).ravel())
            else:
                raise ValueError("No support vectors found. Cannot compute bias term.")
        
    def primal_dual_solver(self,lr=0.01,max_iter=1000,batch_size=32,method='bgd',solver='primal'):
        if solver=='primal':
            self.primal_solver(lr,max_iter,batch_size,method)
        elif solver=='dual':
            self.dual_solver(lr,max_iter,batch_size,method)
        else:
            raise ValueError("solver must be 'primal' or 'dual'")
        
    def duality_gap_score(self):
        if self.w is None or self.b is None or self.alpha is None or self.beta is None:
            raise ValueError("Run primal_dual_solver() first.")
        self.P=1/2*(self.w.T@self.w).item()+self.c*np.sum(np.maximum(0,1-self.y*(self.x@self.w+self.b)))
        self.D=np.sum(self.alpha)-1/2*(self.alpha.T@self.G@self.alpha)
        return (self.P-self.D).item()
    
    def accuracy_score(self):
        if self.w is None or self.b is None:
            raise ValueError("Run primal_dual_solver() first.")
        y_pred=np.sign(self.x@self.w+self.b)
        return np.mean(y_pred==self.y).item()
    
    def predict(self,x):
        if self.w is None or self.b is None:
            raise ValueError("Run primal_dual_solver() first.")
        x=np.array(x)
        if x.ndim==1:
            x=x.reshape(1,-1)
        return np.sign(x@self.w+self.b)
    
    def slack_variables(self):
        if self.w is None or self.b is None:
            raise ValueError("Run primal_dual_solver() first.")
        self.slack_i=np.maximum(0,1-self.y*(self.x@self.w+self.b))
        return self.slack_i
    
    def hinge_loss(self):
        if self.w is None or self.b is None:
            raise ValueError("Run primal_dual_solver() first.")
        return np.mean(np.maximum(0,1-self.y*(self.x@self.w+self.b))).item()
    
    def free_support_vectors(self):
        if self.alpha is None or self.beta is None:
            raise ValueError("Run primal_dual_solver() first.")
        return [i for i in range(self.m) if 1e-6 < self.alpha[i,0] < self.c-1e-6]
    
    def boundary_support_vectors(self):
        if self.alpha is None or self.beta is None:
            raise ValueError("Run primal_dual_solver() first.")
        return [i for i in range(self.m) if self.alpha[i,0]>=self.c-1e-6]
    
    def support_vectors(self):
        free_sv=self.free_support_vectors()
        boundary_sv=self.boundary_support_vectors()
        return free_sv+boundary_sv

    def non_support_vectors(self):
        if self.alpha is None or self.beta is None:
            raise ValueError("Run primal_dual_solver() first.")
        return [i for i in range(self.m) if self.alpha[i,0]<=1e-6]

import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

np.random.seed(42)

# =====================================================
# Moderately Overlapping Dataset
# =====================================================

n_samples_per_class = 150

# Class -1
X1 = (
    np.random.randn(n_samples_per_class, 2) * 1.2
    + np.array([-2.0, -2.0])
)

# Class +1
X2 = (
    np.random.randn(n_samples_per_class, 2) * 1.2
    + np.array([2.0, 2.0])
)

X = np.vstack((X1, X2))

y = np.vstack((
    -np.ones((n_samples_per_class, 1)),
     np.ones((n_samples_per_class, 1))
))

# =====================================================
# Add Some Challenging Outliers
# =====================================================

outliers = np.array([
    [ 4.5,  4.5],
    [ 5.0, -4.0],
    [-5.0,  3.5],
    [-4.5, -4.5],
    [ 3.5, -3.5],
    [-3.5,  3.5],
    [ 5.0,  0.0],
    [ 0.0,  5.0]
])

outlier_labels = np.array([
    [-1],
    [ 1],
    [-1],
    [ 1],
    [ 1],
    [-1],
    [ 1],
    [-1]
])

X = np.vstack((X, outliers))
y = np.vstack((y, outlier_labels))

# =====================================================
# Small Label Noise (3%)
# =====================================================

noise_idx = np.random.choice(
    len(y),
    size=int(0.03 * len(y)),
    replace=False
)

y[noise_idx] *= -1

# =====================================================
# Shuffle Dataset
# =====================================================

idx = np.random.permutation(len(y))

X = X[idx]
y = y[idx]

print("Dataset Shape :", X.shape)
print("Labels Shape  :", y.shape)

# =====================================================
# Visualization
# =====================================================

plt.figure(figsize=(8,6))

plt.scatter(
    X[y[:,0]==-1,0],
    X[y[:,0]==-1,1],
    label="-1",
    alpha=0.7
)

plt.scatter(
    X[y[:,0]==1,0],
    X[y[:,0]==1,1],
    label="+1",
    alpha=0.7
)

plt.title(
    "Moderately Overlapping Dataset with Outliers"
)

plt.xlabel("Feature 1")
plt.ylabel("Feature 2")

plt.grid(True)
plt.legend()
plt.show()

# =====================================================
# Train/Test Split
# =====================================================

X_train,X_test,y_train,y_test = train_test_split(
    X,
    y,
    test_size=0.25,
    random_state=42,
    stratify=y
)

# =====================================================
# Compare Primal vs Dual
# =====================================================

methods=["bgd","sgd","mini-bgd"]

for method in methods:

    print("\n"+"="*70)
    print("METHOD :",method.upper())
    print("="*70)

    model_primal = SoftMargin_dual_primal_system(
        X_train,
        y_train,
        c=1.0
    )

    model_primal.primal_dual_solver(
        lr=0.01,
        max_iter=5000,
        batch_size=16,
        method=method,
        solver='primal'
    )

    primal_w = model_primal.w.copy()
    primal_b = model_primal.b

    train_pred_primal = np.sign(
        X_train @ primal_w + primal_b
    )

    test_pred_primal = np.sign(
        X_test @ primal_w + primal_b
    )

    primal_train_acc = np.mean(
        train_pred_primal == y_train
    )

    primal_test_acc = np.mean(
        test_pred_primal == y_test
    )

    model_dual = SoftMargin_dual_primal_system(
        X_train,
        y_train,
        c=1.0
    )

    model_dual.primal_dual_solver(
        lr=0.001,
        max_iter=5000,
        batch_size=16,
        method=method,
        solver='dual'
    )

    dual_w = model_dual.w.copy()
    dual_b = model_dual.b

    train_pred_dual = np.sign(
        X_train @ dual_w + dual_b
    )

    test_pred_dual = np.sign(
        X_test @ dual_w + dual_b
    )

    dual_train_acc = np.mean(
        train_pred_dual == y_train
    )

    dual_test_acc = np.mean(
        test_pred_dual == y_test
    )

    print("\nPRIMAL")
    print("Train Accuracy :", primal_train_acc)
    print("Test Accuracy  :", primal_test_acc)

    print("\nDUAL")
    print("Train Accuracy :", dual_train_acc)
    print("Test Accuracy  :", dual_test_acc)

    print("\nCOMPARISON")
    print(
        "Weight Difference :",
        np.linalg.norm(primal_w-dual_w)
    )

    print(
        "Bias Difference :",
        abs(primal_b-dual_b)
    )

    print(
        "Duality Gap :",
        model_dual.duality_gap_score()
    )

# =====================================================
# Prediction on New Points
# =====================================================

print("\n"+"="*70)
print("PREDICTION EXAMPLES")
print("="*70)

sample_points=np.array([
    [3,3],
    [-3,-3],
    [0,0],
    [2,-2],
    [-2,2],
    [1,1],
    [-1,-1]
])

pred=model_dual.predict(sample_points)

for point,label in zip(sample_points,pred):

    print(
        f"Point {point} --> "
        f"Predicted Class {int(label[0])}"
    )

# =====================================================
# Decision Boundary
# =====================================================

xx,yy=np.meshgrid(
    np.linspace(X[:,0].min()-1,X[:,0].max()+1,300),
    np.linspace(X[:,1].min()-1,X[:,1].max()+1,300)
)

grid=np.c_[xx.ravel(),yy.ravel()]

Z=model_dual.predict(grid)
Z=Z.reshape(xx.shape)

plt.figure(figsize=(9,7))

plt.contourf(
    xx,
    yy,
    Z,
    alpha=0.25,
    levels=[-1,0,1]
)

plt.scatter(
    X[y[:,0]==-1,0],
    X[y[:,0]==-1,1],
    label="-1",
    alpha=0.7
)

plt.scatter(
    X[y[:,0]==1,0],
    X[y[:,0]==1,1],
    label="+1",
    alpha=0.7
)

plt.title(
    "Soft-Margin SVM Decision Boundary"
)

plt.xlabel("Feature 1")
plt.ylabel("Feature 2")

plt.grid(True)
plt.legend()
plt.show()