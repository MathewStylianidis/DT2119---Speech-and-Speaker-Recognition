import numpy as np
import matplotlib.pylab as plt
import matplotlib.pylab as plt
from sklearn.mixture import GMM

component_numbers = [32] #4,8,16,32]
D = np.load('./data/global_distance_matrix.npy')

utterances = [D[16], D[17], D[38], D[39]]

for component_number in component_numbers:
    model = GMM(component_number)
    model.fit(D )
    cluster_labels = model.predict(D)
    posterior = model.predict_proba(D)

    plt.plot(cluster_labels)
    plt.show()



    plt.plot(posterior[10], color = 'r')
    plt.plot(posterior[11], color = 'r')

    plt.plot(posterior[17])
    plt.plot(posterior[16])

    plt.plot(posterior[39])
    plt.plot(posterior[38])

    plt.plot(posterior[29], color = 'r')
    plt.plot(posterior[30], color = 'r')

    plt.show()

    plt.pcolormesh(posterior)
    plt.grid()
    plt.show()
