<h1 style="color:blue">Analysis of Unrecognized User Requests in Goal-Oriented Dialog Systems</h1>

**Project BY : Dvir Magen, Yuval Vinokur**

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
The number of clustering is unknown and must be discovered by the clustering 
algorithm.  
surfacing topical clusters in unhandled requests is done by implementing the
clustering algorithm that uses the Sentence-Bert (SBERT) model but without defining
constant k, also tried to use the Tf-idf model and get insufficient results.  
At first, we read the input data file and insert the sentences into a list.  Then, encode the 
sentences by using the pre-trained model. 
 
The embeddings are then transformed into a sparse matrix, computes the cosine 
similarity between an input item (a sentence embedding) and each of the centroids, and 
builds a list of similarities and the index of the centroid with the highest similarity.   
Updates the centroids based on the items in each cluster.   
 It creates a list of lists, where each sublist contains the embeddings of the items in a cluster.    
The sublists with fewer than minimum items to cluster that define in the config file are 
removed. For each remaining sublist, the mean of the embeddings is computed, and if it 
is different from the corresponding centroid, the centroid is updated. The new centroids 
and a boolean variable indicating whether the algorithm can stop early. 


We initialize an empty list of centroids and randomly select an item from the input 
data to be the first centroid, then create an empty list of clusters of the same length as 
the input data, where each element is initialized to -1, then shuffles the indexes of the 
input data and iterates through them and updating the clusters and the centroids.
If the algorithm reaches a state where the centroids do not change, it stops early. The 
maximum number of iterations is set to 10.  
By using a similarity threshold of 0.65, the code ensures that only items that are 
sufficiently like an existing cluster centroid are added to that cluster. 
This helps to control the growth of the number of clusters since new clusters are only 
created for items that are not like any existing cluster centroid. Then, filtering the 
clusters to exclude those with fewer than 10 items. The clusters list contains the 
sentences and their corresponding data points grouped into clusters.

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