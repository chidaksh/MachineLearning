import data_preprocess as dp
import gibbs as g
import collapsedGibbs as cg
import numpy as np
import matplotlib.pyplot as plt
# Plotting
i = np.random.choice(np.arange(1,dp.num_docs+1),2)
print(i)
for x in i:
    plt.plot(g.theta[x],label = "GibbsSampler")
    plt.plot(cg.theta[x],label = "CollapsedGibbsSampler")
    plt.title("Topic distribution $theta_i$ for document {}".format(x))
    plt.xlabel("Topic id")
    plt.ylabel("Probability")
    plt.legend()
    plt.show()

inv_vocabulary = {v: k for k, v in dp.vocabulary.items()}
n_top_words = 10
temp = np.transpose(g.cwt)
for topic_idx, topic in enumerate(temp):
    message = "Topic #%d: " % topic_idx
    message += " ".join([inv_vocabulary[i] for i in topic.argsort()[:-n_top_words - 1:-1]])
    print(message)

temp = np.transpose(cg.cwt)
for topic_idx, topic in enumerate(temp):
    message = "Topic #%d: " % topic_idx
    message += " ".join([inv_vocabulary[i] for i in topic.argsort()[:-n_top_words - 1:-1]])
    print(message)