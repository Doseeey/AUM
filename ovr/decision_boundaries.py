import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

def plot_decision_boundary(model, X, y):
    # X = [[f1, f2]...]
    h = .02
    extend_bounds = 1
    x_min, x_max = X[:, 0].min() - extend_bounds, X[:, 0].max() + extend_bounds # +1 to have spare space from the edge
    y_min, y_max = X[:, 1].min() - extend_bounds, X[:, 1].max() + extend_bounds

    #Matrix with columns from x_min to x_max with step h
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                         np.arange(y_min, y_max, h))

    
    Z = model.predict(np.c_[xx.ravel(), yy.ravel()]) #flatten data and merge them back to shape of X_embedded
    Z = np.where(Z == 0, 2, np.where(Z == 2, 0, 1)).reshape(xx.shape)

    #set boundary
    #meshgrid and labels
    plt.contourf(xx, yy, Z, alpha=0.3, cmap='coolwarm')
    
    #x - f1 / y - f2
    scat = plt.scatter(X[:, 0], X[:, 1], c=y, cmap='coolwarm', edgecolors='k')
    plt.legend(*scat.legend_elements())
    plt.title('Decision boundaries')
    plt.show()