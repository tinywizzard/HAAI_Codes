"""
This program finds the best value of K in KMeans algorithm using Silhouette Coefficient for 'housing.csv' dataset. The range of K values to analyze is provided as a command line parameter.
Syntax: python assignment.py <number> <number>

For example, to search best K between 3 and 6 the command line input should be:
python assignment.py 3 6
"""

# importing the libraries

"""  DO NOT MODIFY  """
import sys
import pandas as pd
from sklearn.cluster import KMeans
from sklearn import metrics
"""  DO NOT MODIFY  """

def find_best_kmeans(data, min_k, max_k):

    """  write from here  """

        # Initialize the K-Means model
    # s_dict = dict()
    max_coeff = 0
    best_k = 0
    for i in range(min_k, max_k+1):
        kmeans = KMeans(n_clusters=i, n_init='auto', random_state=0)
        kmeans.fit(data)
        c=kmeans.labels_
        # Use the data and calculate Silhouette Coefficient for the range of K provided
        # s_dict.update({metrics.silhouette_score(data, c, metric='euclidean') : i})
        if max_coeff < metrics.silhouette_score(data, c, metric='euclidean'):
            max_coeff = metrics.silhouette_score(data, c, metric='euclidean')
            best_k = i
        # Return the best K with respect to Silhouette Coefficient 
    # return s_dict[max(s_dict.keys())]
    return best_k
    """        End        """


"""  DO NOT MODIFY  """
if __name__ == '__main__':

    """
    ALERT: * * * No changes are allowed in this section  * * *
    """
 
    if len(sys.argv) == 2:
        print("Usage: python assignment.py <number> <number>")
        sys.exit(1)

    input_data_one = sys.argv[1].strip()
    input_data_two = sys.argv[2].strip()
    
    """  Call to function that will perform the computation. """
    if input_data_one.isdigit() and input_data_two.isdigit():

        min_k = int(input_data_one)
        max_k = int(input_data_two)
        if min_k>=2 or max_k>min_k:
            data =pd.read_csv("./housing.csv")
            print(find_best_kmeans(data, min_k, max_k)) 
        else:
           print("Invalid input")
    else:
        print("Invalid input")
 
    
    """ End to call """