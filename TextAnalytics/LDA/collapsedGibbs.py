import data_preprocess as dp
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

def initialize():
    # model hyperparameters
    global alpha,beta,cwt,cdt,csum,z,theta,phi
    alpha = 50/dp.num_topics
    beta = 0.01

    cwt = np.zeros((dp.num_words, dp.num_topics))
    cdt = np.zeros((dp.num_docs, dp.num_topics))
    csum = np.zeros(dp.num_topics)

    # variables to be infered from the data
    z = [[0 for i in range(len(j))] for j in dp.docs]
    # probability dist over topics for all documents
    theta = np.zeros((dp.num_docs, dp.num_topics))
    # probability dist over words for all topics
    phi = np.zeros((dp.num_topics, dp.num_words))
    for i, doc in enumerate(dp.docs):
        for j, word in enumerate(doc):
            z[i][j] = j % dp.num_topics
            topic = z[i][j]
            cdt[i][topic] += 1
            cwt[word][topic] += 1
    print("intialization done")
    return

# Collapsed Gibbs Sampler
def CollapsedGibbsSampler():
    global aplha,beta
    conditional_prob = np.zeros(dp.num_topics)
    for iteration in tqdm(range(500), desc="Progress Bar"):
        for i, doc in enumerate(dp.docs):
            for j, word in enumerate(doc):
                topic = z[i][j]
                cdt[i][topic] -= 1
                cwt[word][topic] -= 1
                conditional_prob[topic] = ((cdt[i][topic] + alpha) * (beta + cwt[word][topic])) / ((dp.num_topics * alpha + np.sum(cdt[i,:])) * (dp.num_words*beta + np.sum(cwt[:,topic])))                
                conditional_prob /= np.sum(conditional_prob)
                dist = np.random.multinomial(1, conditional_prob)
                new_topic = np.argmax(dist, axis=0)
                z[i][j] = new_topic
                cdt[i][new_topic] += 1
                cwt[word][new_topic] += 1
    for i, doc in enumerate(dp.docs):
        for j, word in enumerate(doc):
            topic = z[i][j]
            phi[topic][word] = (cwt[word][topic] + beta)/(dp.num_words * beta + np.sum(cwt[:,topic])) 
            theta[i][topic] = (cdt[i][topic] + alpha)/(dp.num_topics * alpha + np.sum(cdt[i,:]))
        
    return

initialize()
CollapsedGibbsSampler()