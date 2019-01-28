import numpy as np
import multiprocessing
import imageio
import scipy.ndimage
import skimage.color
import sklearn.cluster
import scipy.spatial.distance
import os,time
import util
import random
import skimage.io

def extract_filter_responses(image):
    '''
    Extracts the filter responses for the given image.

    [input]
    * image: numpy.ndarray of shape (H,W) or (H,W,3)
    [output]
    * filter_responses: numpy.ndarray of shape (H,W,3F)
    '''
    # If the image is gray scale
    if len(image.shape) == 2:
        image = np.tile(image[:, newaxis], (1, 1, 3))

    if image.shape[2] == 4:
        image = image[:,:,0:3]

    image = skimage.color.rgb2lab(image)
    
    scales = [1,2,4,8,8*np.sqrt(2)]
    for i in range(len(scales)):
        for c in range(3):
            #img = skimage.transform.resize(image, (int(ss[0]/scales[i]),int(ss[1]/scales[i])),anti_aliasing=True)
            img = scipy.ndimage.gaussian_filter(image[:,:,c],sigma=scales[i])
            if i == 0 and c == 0:
                imgs = img[:,:,np.newaxis]
            else:
                imgs = np.concatenate((imgs,img[:,:,np.newaxis]),axis=2)
        for c in range(3):
            img = scipy.ndimage.gaussian_laplace(image[:,:,c],sigma=scales[i])
            imgs = np.concatenate((imgs,img[:,:,np.newaxis]),axis=2)
        for c in range(3):
            img = scipy.ndimage.gaussian_filter(image[:,:,c],sigma=scales[i],order=[0,1])
            imgs = np.concatenate((imgs,img[:,:,np.newaxis]),axis=2)
        for c in range(3):
            img = scipy.ndimage.gaussian_filter(image[:,:,c],sigma=scales[i],order=[1,0])
            imgs = np.concatenate((imgs,img[:,:,np.newaxis]),axis=2)

    return imgs

def get_visual_words(image,dictionary):
    '''
    Compute visual words mapping for the given image using the dictionary of visual words.

    [input]
    * image: numpy.ndarray of shape (H,W) or (H,W,3)
    
    [output]
    * wordmap: numpy.ndarray of shape (H,W)
    '''
    
    # ----- TODO -----
    pass


def compute_dictionary_one_image(args):
    '''
    Extracts random samples of the dictionary entries from an image.
    This is a function run by a subprocess.

    [input]
    * i: index of training image
    * alpha: number of random samples
    * image_path: path of image file
    * time_start: time stamp of start time

    [saved]
    * sampled_response: numpy.ndarray of shape (alpha,3F)    '''

    i,alpha,image_path = args
    image = skimage.io.imread(image_path)
    response = extract_filter_responses(image)
    x = np.random.choice(response.shape[0], alpha)
    y = np.random.choice(response.shape[1], alpha)
    sampled_response = response[x, y, :]
    np.savez("../tmp/" + str(i) + ".npz", responses = sampled_response)
    # ----- TODO -----
    #pass

def compute_dictionary(num_workers=2):
    '''
    Creates the dictionary of visual words by clustering using k-means.

    [input]
    * num_workers: number of workers to process in parallel
    
    [saved]
    * dictionary: numpy.ndarray of shape (K,3F)
    '''

    train_data = np.load("../data/train_data.npz")
    data_num = len(train_data['files'])
    alpha = 50
    K = 100
    filter_num = 20
    directory = "../tmp"
    if not os.path.exists(directory):
        os.makedirs(directory)
    
    #if __name__ == '__main__':
    with multiprocessing.Pool(num_workers) as p:
        args = zip(list(range(data_num)), [alpha] * data_num, train_data['files'])
        p.map(compute_dictionary_one_image, args)
    
    filter_responses = np.empty([0, filter_num * 3])        
    for i in range(data_num):
        data = np.load("../tmp/" + str(i) + ".npz")
        filter_responses = np.append(filter_responses, data['responses'], axis = 0)
        
    kmeans = sklearn.cluster.KMeans(n_clusters = K, n_jobs = num_workers).fit(filter_responses)
    dictionary = kmeans.cluster_centers_
    np.save("dictionary.npy", dictionary)    
    # ----- TODO -----
    #pass


