import numpy as np
import matplotlib.pyplot as plt
from ucimlrepo import fetch_ucirepo 
from sklearn.manifold import TSNE
from Perceptron import Perceptron
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

def plot_decision_boundary(model, X, y):
    # X = [[f1, f2]...]
    h = .02
    extend_bounds = 5
    x_min, x_max = X[:, 0].min() - extend_bounds, X[:, 0].max() + extend_bounds # +1 to have spare space from the edge
    y_min, y_max = X[:, 1].min() - extend_bounds, X[:, 1].max() + extend_bounds

    #Matrix with columns from x_min to x_max with step h
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                         np.arange(y_min, y_max, h))

    
    Z = model.predict(np.c_[xx.ravel(), yy.ravel()]) #flatten data and merge them back to shape of X_embedded
    Z = Z.reshape(xx.shape)
    
    #set boundary
    #meshgrid and labels
    plt.contourf(xx, yy, Z, alpha=0.3, cmap='coolwarm')
    
    #x - f1 / y - f2
    scat = plt.scatter(X[:, 0], X[:, 1], c=y, cmap='coolwarm', edgecolors='k')
    plt.legend(*scat.legend_elements())
    plt.title('Decision boundary of Banknote Authentication Classification')
    plt.show()

# fetch dataset 
banknote_authentication = fetch_ucirepo(id=267) 
X = np.array(banknote_authentication.data.features, dtype=float) 
y = np.array(banknote_authentication.data.targets, dtype=int) 

scaler = StandardScaler()
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

X_embedded = TSNE(n_components=2, random_state=42).fit_transform(scaler.fit_transform(X))

perceptron_tsne = Perceptron(learning_rate=0.01, n_iter=1000)
perceptron_tsne.fit(X_embedded, y)
plot_decision_boundary(perceptron_tsne, X_embedded, y)