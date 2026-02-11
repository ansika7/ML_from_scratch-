import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
df=pd.read_csv(r"C:\Users\ansik\Downloads\archive (22)\car details v4.csv")
df.info()
df["Engine"].value_counts() 
df["Max Power"].value_counts()
df["Max Torque"].value_counts()
df["Drivetrain"].value_counts()
df["Engine"]=df["Engine"].fillna("1197cc")
df["Max Power"]=df["Max Power"].fillna("89 bhp @ 4000 rpm")
df["Max Torque"]=df["Max Torque"].fillna("200 Nm @ 1750 rpm ")
df["Drivetrain"]=df["Drivetrain"].fillna("FWD")
df["Length"]=df["Length"].fillna(np.mean(df["Length"]))
df["Width"]=df["Width"].fillna(np.mean(df["Width"]))
df["Height"]=df["Height"].fillna(np.mean(df["Height"]))
df["Seating Capacity"]=df["Seating Capacity"].fillna(np.mean(df["Seating Capacity"]))
df["Fuel Tank Capacity"]=df["Fuel Tank Capacity"].fillna(np.mean(df["Fuel Tank Capacity"]))
col = df.columns
dt = df.dtypes.values
for i in range(len(col)):
    if(dt[i] == 'object'):
        df[col[i]] = df[col[i]].astype('category').cat.codes
for i in range(len(col)):
    if(dt[i] != 'object'):
        X=df[col[i]]
        df[col[i]]=(X-np.mean(X))/np.std(X)
X=df
n=X.shape[0]
cov_matrix=np.dot(X.T,X)/(n-1)
def power_iteration(A, iterations=1000, tol=1e-6):
    n = A.shape[0]
    b = np.random.rand(n)
    #b = b / np.linalg.norm(b)
    norm_b=(np.dot(b.T,b))**(0.5)
    b=b/norm_b

    for _ in range(iterations):
        b_new = A @ b
        b_new = b_new / np.linalg.norm(b_new)

        if np.linalg.norm(b - b_new) < tol:
            break
        b = b_new

    eigenvalue = b.T @ A @ b
    return eigenvalue, b
def deflate(A, eigenvalue, eigenvector):
    return A - eigenvalue * np.outer(eigenvector, eigenvector)
def manual_eigen_decomposition(A, k):
    A_copy = A.copy()
    eigenvalues = []
    eigenvectors = []

    for _ in range(k):
        val, vec = power_iteration(A_copy)
        eigenvalues.append(val)
        eigenvectors.append(vec)
        A_copy = deflate(A_copy, val, vec)

    return np.array(eigenvalues), np.array(eigenvectors).T
k = 2
eigenvalues, eigenvectors = manual_eigen_decomposition(cov_matrix, k)

print("Eigenvalues:", eigenvalues)

X_pca = X @ eigenvectors
print("Reduced data shape:", X_pca.shape)
X_pca=np.array(X_pca)

plt.figure(figsize=(8,6))
plt.scatter(X_pca[:,0], X_pca[:,1], alpha=0.6)
plt.xlabel("Principal Component 1")
plt.ylabel("Principal Component 2")
plt.title("PCA from Scratch on Vehicle Dataset")
plt.show()
