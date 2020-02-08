import utils.general as utils

import utils.gt.io_utils as gtio
import utils.gt.analysis_utils as gtanalysis

#datadir='~/mowgli-in-the-jungle/kb/'
datadir='/nas/home/ilievski/mowgli-in-the-jungle/kb/'
#datadir='/Users/filipilievski/mowgli-dev-framework/kb/'
mowgli_nodes=f'{datadir}nodes_v002.csv'
mowgli_edges=f'{datadir}edges_v002.csv'
output_gml=f'{datadir}graph.graphml'

g=gtio.load_gt_graph(output_gml.replace(".graphml", '.gt'))

print('graph loaded. now computing centrality.')
node_pagerank=gtanalysis.compute_pagerank(g)
print('ppagerank computed')
hits=gtanalysis.compute_hits(g)
print('hits computed')
bt=gtanalysis.compute_betweenness(g)
print('bt computed')
