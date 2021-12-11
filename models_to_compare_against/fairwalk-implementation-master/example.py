import numpy as np
import networkx as nx
from fairwalk import FairWalk

# FILES
EMBEDDING_FILENAME = './credit_noise.emb'
EMBEDDING_MODEL_FILENAME = './credit_noise.model'

# Create a graph
#graph = nx.fast_gnp_random_graph(n=100, p=0.5)
graph = nx.read_edgelist('graph_credit_noise.edgelist', delimiter = ',', nodetype = int)

n = len(graph.nodes())
node2group = {node: group for node, group in zip(graph.nodes(), (5*np.random.random(n)).astype(int))}
nx.set_node_attributes(graph, node2group, 'group')

# Precompute probabilities and generate walks
model = FairWalk(graph, dimensions=64, walk_length=10, num_walks=40, workers=1) #walk_length =30, num_walks =200

## if d_graph is big enough to fit in the memory, pass temp_folder which has enough disk space
# Note: It will trigger "sharedmem" in Parallel, which will be slow on smaller graphs
# fairwalk = FairWalk(graph, dimensions=64, walk_length=30, num_walks=200, workers=4, temp_folder="/mnt/tmp_data")

# Embed
model = model.fit(window=10, min_count=1,
                  batch_words=4)  # Any keywords acceptable by gensim.Word2Vec can be passed, `diemnsions` and `workers` are automatically passed (from the FairWalk constructor)

# Look for most similar nodes
model.wv.most_similar('2')  # Output node names are always strings

# Save embeddings for later use
model.wv.save_word2vec_format(EMBEDDING_FILENAME)

# Save model for later use
model.save(EMBEDDING_MODEL_FILENAME)
