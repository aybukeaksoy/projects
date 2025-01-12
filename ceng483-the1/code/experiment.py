import os
import cv2
from color_histogram import *

def instance_recognition(query_number, n_by_n, color_histogram_type, color_space_type, quantization_interval):

    support_folder_path = 'dataset/support_96'
    query_folder_path = 'dataset/query_1'
    if query_number == 2:
        query_folder_path = 'dataset/query_2'
    elif query_number == 3:
        query_folder_path = 'dataset/query_3'
    support_histograms_cache={}
    max_similarity=0
    max_index=-1
    index=0
    correctly_predicted=0
    for support_image in os.listdir(support_folder_path):
        s_image = cv2.imread(os.path.join(support_folder_path, support_image))
        support_histograms_cache[support_image]=grid_based_feature_extraction(s_image,n_by_n,color_histogram_type,color_space_type,quantization_interval)
    for query_image in os.listdir(query_folder_path):
        q_image = cv2.imread(os.path.join(query_folder_path, query_image))
        query_histograms=grid_based_feature_extraction(q_image,n_by_n,color_histogram_type,color_space_type,quantization_interval)
        for support_image in support_histograms_cache:
            similarity = np.sum(np.sum(np.minimum(query_histograms,support_histograms_cache[support_image]),axis=1))/len(query_histograms)
            if similarity>max_similarity:
                max_index=support_image
                max_similarity=similarity
            index+=1
        max_similarity = 0
        index = 0
        if query_image==max_index:
            correctly_predicted+=1
    accuracy=correctly_predicted/200
    return accuracy
