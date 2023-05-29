# <font color ='darkorange'> **High Value Customers Identification ( Insiders )**</font>

## <font color ='darkorange'> 0.0. Solution Planning (IOT) </font>

### <font color ='darkorange'>Input - Entrada </font>

1.Business Problem
- Selecting the most valuable customers to join a Loyalty Program.

2. Dataset
 - Sales data from an online e-commerce platform, covering a one-year period.

### <font color ='darkorange'>Ouput - Saida</font>

1. Identification of individuals to be part of the Insiders program

 - List:
 
    |client_id | is_insiders |
    | -------- | ----------- |
    |10323     | yes/1       |
    |32413     | no/0        |

2. Report with answers to business questions:
    1. Who are the eligible individuals to participate in the Insiders program?
    2. How many customers will be included in the group?
    3. What are the key characteristics of these customers?
    4. What is the percentage of revenue contribution from the Insiders?


### <font color ='darkorange'>Task</font>

**1. Who are the eligible individuals to participate in the Insiders program?**
- What does it mean to be eligible? What are high-value customers?
    - Revenue:
        - High average ticket.
        - High customer lifetime value (LTV).
        - Low recency.
        - High basket size.
        - Low churn probability. - models
        - High LTV prediction. - models
        - High purchase propensity. - models
    - Cost:
        - Low return rate.
            
**2. How many customers will be included in the group?**
    - Total number of customers
    - % of the Insiders group

**3. What are the key characteristics of these customers?**
    - Write customer characteristics:
        - Age
        - Location
    - Write consumption characteristics:
        - Cluster attributes.

**4. What is the percentage of revenue contribution from the Insiders?**
    - Total revenue for the year
    - Total revenue from the Insiders group for the year

## <font color ='darkorange'> 1.0. The Project:</font>
Assumptions:

invoices_no - Invoices with C, meaning Charge back

## <font color ='darkorange'>1.1. CRISP Cycles</font>

### <font color ='darkorange'> C01 - RFM</font>
As the company lacks any customer segmentation, I want to quickly aggregate value to their business. In the first cycle, I used the RFM algorithm, which segments customers based on Recency, Frequency, and Monetary (gross revenue). RFM is a basic algorithm that is easy to implement and creates high value for the company.

### <font color ='darkorange'> C02 - Baseline</font>

In the second cycle, the objective is to create a basic clustering algorithm, and define the fundamental metrics, and tools. The main goal is to establish a baseline for future cycles and compare the results with the baseline. It is also crucial to communicate the progress of the project to stakeholders, manage their expectations, and keep them informed of any significant developments.

- Added new features.

- Optimized KMeans clustering algorithm by using KMeans++
- Performed Silhouette Analysis to determine the optimal number of clusters
- Created a bivariate plot using pair plot to visualize relationships between variables
- Used UMAP for data visualization and dimensionality reduction

### <font color ='darkorange'> C03 - Improving the baseline</font>
In the third cycle, the primary objective is to enhance the metrics used in the second cycle. To achieve this, various techniques are employed, including dimensionality reduction, latent spaces (embedding), and the exploration of new algorithms. The following techniques are implemented:
- Principal Component Analysis (PCA)
- t-SNE (t-Distributed Stochastic Neighbor Embedding)
- UMAP (Uniform Manifold Approximation and Projection)
- Tree-Based Embedding

In addition to these techniques, several clustering algorithms are tested to improve the clustering performance. The algorithms used in this phase include:
- KMeans
- Gaussian Mixture Models (GMM)
- Hierarchical Clustering
- DBSCAN (Density-Based Spatial Clustering of Applications with Noise)

By employing these techniques and algorithms, the aim is to refine the clustering results, discover intricate patterns, and uncover valuable insights from the data.

## <font color ='darkorange'> 2.0. Searchs:</font>
### Algorithms
KMeans, KMeans++, KMesoides - https://towardsdatascience.com/understanding-k-means-k-means-and-k-medoids-clustering-algorithms-ad9c9fbf47ca

KMeans ++ - https://avichawla.substack.com/p/this-small-tweak-can-significantly

DBScan - https://towardsdatascience.com/how-to-use-dbscan-effectively-ed212c02e62

DBScan - https://towardsdatascience.com/machine-learning-clustering-dbscan-determine-the-optimal-value-for-epsilon-eps-python-example-3100091cfbc

DBScan - https://www.datanovia.com/en/lessons/dbscan-density-based-clustering-essentials/

DBScan - https://www.kdnuggets.com/2020/04/dbscan-clustering-algorithm-machine-learning.html

DBScan - https://medium.com/@tarammullin/dbscan-parameter-estimation-ff8330e3a3bd

### t-SNE & UMAP
t-SNE ( StatQuest ) - https://www.youtube.com/watch?v=NEaUSP4YerM

UMAP ( StatQuest ) - https://www.youtube.com/watch?v=eN0wFzBA4Sc

UMAP ( AI Coffe Break ) - https://www.youtube.com/watch?v=6BPl81wGGP8&t=177s

Understanding UMAP - https://pair-code.github.io/understanding-umap/

UMAP Paper - https://arxiv.org/abs/1802.03426

Tree Based Embedding - https://gdmarmerola.github.io/forest-embeddings/

### Distances

cossenos - https://demacdolincoln.github.io/anotacoes-nlp/posts/distancia-euclidiama-vs-similaridade-de-cossenos/

## Research Desk
https://www.kaggle.com/code/marcinrutecki/clustering-methods-comprehensive-study

https://www.kaggle.com/code/hasanbasriakcay/e-commerce-forecasting-fbprophet-optuna/notebook

https://scikit-learn.org/stable/auto_examples/cluster/plot_cluster_comparison.html#sphx-glr-auto-examples-cluster-plot-cluster-comparison-py