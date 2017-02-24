# Content Based Image Retriever

Given an input image, finds similar images in a given database.
For faster results the database should be divided by categories.

The program pre-processes the images in the database and caches the results for faster image retrieval.

This content-based image retrieval works in two steps.  
First we classify the image using SVM's trained with its
bag of words (BoW) representation and, optionally, color histograms.  
Then we search for similar images in the predicted category using the cosine similarity index and
refine that search using spatial verification with RANSAC.