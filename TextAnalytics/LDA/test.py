import numpy as np
import data_preprocess as dp
print(np.argmax(np.random.multinomial(100, [1/dp.num_topics]*dp.num_topics, size=1)/100))
a = np.array([[1,2,3],[4,5,6],[7,8,9]])
print("sum is: ", np.sum(a,axis = 0))
print(a[:,1])
print(a[1,:])
print(np.sum(a[:,1]))
print(np.sum(a[1,:]))
i = int(np.random.choice(np.arange(1,dp.num_topics+1),1))
print(i)
# print(np.transpose(a))
# c = [0,1]
# print(a[c])
# print(np.linspace(0,100,10))
a = [0.1,0.6,0.9]
# print([int(item * 10) -1 for item in a])
print("args:",np.argmax(a,axis=0))
print("checking multinomial:")
for i in range(10):
    s = np.random.multinomial(1,[0.1,0.02,0.08,0.8])
    print(np.argmax(s,axis = 0),s)
b = np.array([[1,2,4],[2,3,6]])
print(np.sum(b[:,2]))
# import math
# def gamma1(x):
#     return math.exp(-math.log(0.76 + (0.24 * x)) + (1 - x) * ((0.76/(0.76 + 0.24*x) - 0.45/(1 - 0.55*x))))
# def gamma2(x):
#     return math.exp(-math.log(1 - 0.55*x) - x * ((0.76/(0.76 + 0.24*x) - 0.45/(1 - 0.55*x))))
# x = np.arange(0,11)/10
# for i in x:
#     print(f"value of i is {i} and value of gamma1({i}) is {gamma1(i)} and value of gamma2({i}) is {gamma2(i)}")
# import matplotlib.pyplot as plt
# fig, ax = plt.subplots()
# # prime = [0.0237,0.0502,0.0457,0.0421]
# # fibo = [0.0229,0.0586,0.0531,0.0485]
# # palin = [0.0231,0.0695,0.0611,0.0545]
# # even = [0.0224,0.0215,0.0208,0.0201]
# # desc = [0.0229,0.0884,0.078,0.0697]
# # size = [16,128,512,1024]

# prime = [0.0421,0.0421,0.0421,0.0421]
# fibo = [0.0491,0.0488,0.0486,0.0485]
# palin = [0.0545,0.0545,0.0545,0.0545]
# even = [0.0201,0.0201,0.0201,0.0201]
# desc = [0.0624,0.0707,0.0702,0.0697]
# size = [16,128,512,1024]

# pos = ax.get_position()
# ax.set_position([pos.x0, pos.y0, pos.width, pos.height * 0.85])
# ax.plot(size,prime,label = "prime.asm")
# ax.plot(size,fibo,label = "fibonacci.asm")
# ax.plot(size,palin,label = "palindrome.asm")
# ax.plot(size,even,label = "even-odd.asm")
# ax.plot(size,desc,label = "descending.asm")
# ax.set_xlabel("L1i-cache size")
# ax.set_ylabel("IPC")
# ax.legend(
#     loc='upper center', 
#     bbox_to_anchor=(0.5, 1.35),
#     ncol=3, 
# )
# plt.show()


# def collapsed_gibbs():
#     for iteration in tqdm(range(10), desc="Progress Bar"):
#         for i, doc in enumerate(dp.docs):
#             for j, word in enumerate(doc):
#                 topic = z[i][j]
#                 # not including the current instance ??
#                 cdt[i][topic] -= 1
#                 cwt[word][topic] -= 1
#                 # dist of words for a topic is calculated as count of each word
#                 #  with that topic divided by total count of all words for the same topic
#                 csum = np.sum(cwt, axis=0)
#                 # print(csum.shape)
#                 # print(cwt[word][:].shape)
#                 # print(phi[:,word].shape)
#                 phi[:, word] = (cwt[word][:] + beta) / \
#                     (csum + (dp.num_words * beta))
#                 # dist of topics for a document is calculated as count of each topic within that
#                 # document divided by total count of all topics for the same document
#                 theta[i][topic] = (cdt[i][topic] + alpha) / \
#                     (np.sum(cdt[i, :]) + (dp.num_topics * alpha))
#                 conditional_prob = phi[:, word] * theta[i][topic]
#                 if (i == 1 and j == 1):
#                     print(phi[topic][word], theta[i][topic])    
#                     print(conditional_prob.shape)
#                 conditional_prob /= np.sum(conditional_prob)
#                 # sample new topic from this conditional probability
                
#                 if not np.all(conditional_prob > 0):
#                     print(conditional_prob)
#                 dist = np.random.multinomial(1, conditional_prob)
#                 new_topic = np.argmax(dist, axis=0)
#                 z[i][j] = new_topic
#                 cdt[i][new_topic] += 1
#                 cwt[word][new_topic] += 1
#     return
