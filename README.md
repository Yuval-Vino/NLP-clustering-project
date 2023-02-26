<h1 style="color:blue">Analysis of Unrecognized User Requests in Goal-Oriented Dialog Systems</h1>

**Project BY : Dvir Magen, [Yuval Vinokur](https://github.com/Yuval-Vino)**

[Explore our in-depth Colab notebook for a detailed examination of text cluster analysis.](https://colab.research.google.com/drive/1fWjQWfjHEPMp4RGYw_usOFfON4NwJIfP?usp=sharing)

## • Introduction:  
recognize the intent of natural language requests are identified by using intent classifier 
uncertainty – requests that are predicted to have a level of confidence below a certain 
threshold are reported as unrecognized (or unhandled). For example, “How does
quarantine work” request classify as “quarantine work”.  
 Our approach to accomplishing this mission is:  

(1) surfacing topical clusters in unhandled requests (clustering)   
(2) extraction of cluster representatives  
(3) cluster naming (labeling).

<img src="https://i.imgur.com/Ix8tfl3.png" alt="example image" width="400px">


## • Clustering requests:  
Our algorithm leverages the Sentence-Bert (SBERT) model to cluster input data into groups of an unknown number.  
This is achieved by implementing a clustering algorithm that doesn't rely on a fixed k value.   
Our previous attempt to use the Tf-idf model didn't yield satisfactory results.


Initially, we read the input data and create a list of its sentences. Then, using the pre-trained SBERT model, we encode these sentences into embeddings.  
Finally, we use the clustering algorithm to surface topical clusters in unhandled requests.


The algorithm transforms the embeddings into a sparse matrix and calculates the cosine similarity between each sentence embedding and the centroids.  
It generates a list that includes the similarities and the index of the centroid with the highest similarity.  
The algorithm updates the centroids based on the items in each cluster. 
A list of lists is created, with each sublist containing the embeddings of the items in a particular cluster.  
The sublists that contain fewer items than the minimum specified in the configuration file are removed.  
For each remaining sublist, the mean of the embeddings is calculated.   
If the mean is different from the corresponding centroid, the centroid is updated accordingly.  
The algorithm generates new centroids and a boolean variable indicating whether it can stop early.


To start, an empty list of centroids is initialized, and a random item from the input data is selected as the first centroid.  
Additionally, an empty list of clusters is created with the same length as the input data, where each element is initialized to -1.  
The indexes of the input data are shuffled, and the algorithm iterates through them, updating the clusters and the centroids.  
If the centroids do not change, the algorithm stops early, with a maximum of 10 iterations.  
To prevent the growth of the number of clusters, a similarity threshold of 0.65 is used to ensure that only items that are sufficiently similar to an existing cluster centroid are added to that cluster.  
Clusters with fewer than 10 items are filtered out.   
The resulting clusters list contains the sentences and their corresponding data points grouped into clusters based on their similarity to the centroids.

![IDEA](https://imgur.com/3XnokxD.png)

## • Extraction of cluster representatives:

To identify the most representative sentences for each cluster, we use cosine similarity between the centroid of the cluster and each sentence within it.  
This is achieved by utilizing two lists - a list of lists that stores the cosine similarity values between each sentence in a cluster and the cluster's centroid, and a list of lists that stores the actual sentences in each cluster.

Our code calculates the cosine similarity between the cluster's centroid and each sentence in the cluster for each cluster.  
The similarity value is then rounded to four decimal places and stored in the corresponding list along with the sentence itself.  
By selecting the sentences with the highest cosine similarity to the centroid, we can effectively determine the most representative sentences for each cluster.

![Reps](https://imgur.com/UIzlkOR.png)

## • Cluster naming (Labeling):
To prepare the text data for analysis, several preprocessing steps are necessary. These include tokenizing the text, removing stop words, and lemmatizing the words. Once this is done, we can create a corpus of documents and a dictionary of terms with their frequency counts for each cluster of sentences. To train an LDA model with one topic, we need to convert the corpus into a bag-of-words representation.

Using the LDA model, we can identify the most probable words for the topic.  
To further improve our analysis, we should get the n-grams (i.e., sequences of n words) for each sentence in the cluster and use them to create a dictionary that maps each n-gram to its frequency count in the cluster.  
This information can then be used to compute a score for each n-gram in each cluster based on the frequency count and the weight assigned to each term in the topic identified by the LDA model.

Finally, we can choose the n-gram with the highest score of each cluster as the name of the cluster. By doing so, we will have a clear and descriptive name for each cluster that accurately reflects its content. 
Overall, these preprocessing steps and analysis techniques will enable us to gain valuable insights from our text data and make informed decisions based on our findings.

![Cluster Names](https://imgur.com/j7rrDLZ.png)

## • Now that all the individual tasks have been completed, it's time to combine them.
The next step is to create a dictionary for each cluster, containing the cluster name, cluster representatives, and cluster sentences.   
Since we have already identified the representatives of each cluster, we can simply store them in the dictionary along with their corresponding cluster names and original sentences. 
By doing this, we will have a clear overview of each cluster and its contents, allowing us to further analyze and draw insights from our data.  
Additionally, we should also include any unclustered sentences in the same dictionary as the clustered sentences to ensure that our analysis is as comprehensive as possible. This approach will allow us to better understand our data and make more informed decisions based on our findings.

## • Evaluation of the clustering outcome against the provided solution – RI, ARI
The RI and ARI scores were computed using the provided compare_clustering_solutions.py script. To obtain these scores, the input data was partitioned by implementing a clustering algorithm that utilized the Sentence-Bert (SBERT) model. 
Unlike some clustering algorithms, this one did not require a constant number of clusters to be defined. The algorithm calculated the cosine similarity between the cluster's centroid and each sentence in the cluster, rounding the similarity value to 4 decimal places. In addition to this, the LDA model was used to identify the most probable words for the topic, and the score for each n-gram in each cluster was computed based on its frequency count. 
This approach significantly improved the RI and ARI scores.  
![Data scores](https://imgur.com/vrmp3qG.png)
