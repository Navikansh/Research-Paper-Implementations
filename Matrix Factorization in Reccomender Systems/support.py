import numpy as np

def Matrix_factorization(ratings, num_factors=20, steps=100, alpha=0.01, beta=0.02):
    """
    ratings: numpy array of shape (num_users, num_items)
    num_factors: number of latent factors
    steps: number of iterations
    alpha: learning rate
    beta: regularization parameter
    Returns: P, Q matrices"""

    num_users, num_items = ratings.shape

    P = np.random.normal(scale=1./num_factors, size=(num_users, num_factors))
    Q = np.random.normal(scale=1./num_factors, size=(num_items, num_factors))
    rmse_list = []

    #Loop for SGD
    for step in range(steps):
        for u in range(num_users):
            for i in range(num_items):
                if ratings[u, i] > 0:
                    eui = ratings[u, i] - np.dot(P[u, :], Q[i, :].T)
                    P[u, :] += alpha * (eui * Q[i, :] - beta * P[u, :])
                    Q[i, :] += alpha * (eui * P[u, :] - beta * Q[i, :])
        # Calculate the error
        loss = 0
        count = 0
        error_squared_sum = 0
        
        for u in range(num_users):
            for i in range(num_items):
                if ratings[u, i] > 0:
                    loss += (ratings[u, i] - np.dot(P[u, :], Q[i, :].T)) ** 2
                    loss += beta * (np.sum(P[u, :] ** 2) + np.sum(Q[i, :] ** 2))
                    prediction = np.dot(P[u, :], Q[i, :].T)
                    error_squared_sum += (ratings[u, i] - prediction) ** 2
                    count += 1
        rmse = np.sqrt(error_squared_sum / count)
        rmse_list.append(rmse)
        
        if (step + 1) % 50 == 0:
            print(f'Step {step + 1}/{steps}, Loss: {loss:.4f}')
    return P, Q.T, rmse_list



