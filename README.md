# molecular-smile-classification-with-GNN


This task was carried out to predict the activity through molecular structure analysis. If SMILES data is given, it is classified. Machine learning algorithms like RandomForest could be used, but deep learning was used to achieve higher performance. At first, I thought of it as time series data and used a structure such as lstm or transformer, but the performance was not good. Upon further analysis, it was found that the SMILE can be converted into a matrix in the form of a molecular structure. Since the molecular structure is similar to a graph, a graph-based network was used in the project.
