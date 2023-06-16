import data_preprocess as dp
import numpy as np
import matplotlib.pyplot as plt
import collapsedGibbs as cg
from tqdm import tqdm

# Collapsed Gibbs Sampler
def initialise():
    global z,cdt,cwt,theta
    
    cwt = np.zeros((dp.num_words,dp.num_topics))
    cdt = np.zeros((1, dp.num_topics))

    # variables to be infered from the data
    z = [0 for i in range(words)]
    # probability dist over topics for all documents
    theta = np.zeros((1, dp.num_topics))
    # probability dist over words for all topics
    # phi = np.zeros((dp.num_topics, dp.num_words))
    for j, word in enumerate(new_doc):
        # print(word)
        z[j] = j % dp.num_topics
        topic = z[j]
        cdt[0][topic] += 1
        cwt[int(dp.vocabulary[word])][topic] += 1
        
    print("intialization done")
    return

def ClassifyDocUsingCollapsed():
    global z,cdt,cwt,theta
    conditional_prob = np.zeros(dp.num_topics)
    topic_dist = np.zeros(dp.num_topics)
    for iteration in tqdm(range(250), desc="Progress Bar"):
        for j, word in enumerate(new_doc):
            topic = z[j]
            cdt[0][topic] -= 1
            cwt[int(dp.vocabulary[word])][topic] -= 1
            conditional_prob[topic] = ((cdt[0][topic] + cg.alpha) * (cg.beta + cwt[int(dp.vocabulary[word])][topic] + cg.cwt[int(dp.vocabulary[word])][topic])) / ((dp.num_topics * cg.alpha + np.sum(cdt[0,:])) * (dp.num_words * cg.beta + np.sum(cwt[:,topic]) + np.sum(cg.cwt[:,topic])))                
            conditional_prob /= np.sum(conditional_prob)
            dist = np.random.multinomial(1, conditional_prob)
            new_topic = np.argmax(dist, axis=0)
            z[j] = new_topic
            cdt[0][new_topic] += 1
            cwt[int(dp.vocabulary[word])][new_topic] += 1
        
    for j, word in enumerate(new_doc):
        topic = z[j]
        topic_dist[topic] = cg.phi[topic][int(dp.vocabulary[word])]
        # phi[topic][word] = (cwt[word][topic] + beta)/(dp.num_words * beta + np.sum(cwt[:,topic])) 
        theta[0][topic] = (cdt[0][topic] + cg.alpha)/(dp.num_topics * cg.alpha + np.sum(cdt[0,:]))
    
    # print(theta)
    topic_dist /= np.sum(topic_dist)
    for i in range(dp.num_topics):
        print(i,topic_dist[i])
    plt.plot(theta[0])
    plt.xlabel("Topic id")
    plt.ylabel("Probability")
    plt.title("Topic Distribution for the new document")
    plt.show()
    return

ind = np.random.randint(0,dp.num_words,200)
vocab_keys = list(dp.vocabulary.keys())
new_doc = [vocab_keys[i] for i in ind]
words = len(new_doc)
initialise()
ClassifyDocUsingCollapsed()