import networkx as nx


def compute_kg_nsc(G_gt, G_pred):
   """
   Node Set Coverage
   """
   gt_nodes = set(G_gt.nodes())
   pred_nodes = set(G_pred.nodes())


   if len(gt_nodes) == 0:
       return 0


   return len(gt_nodes.intersection(pred_nodes)) / len(gt_nodes)


def compute_kg_ams(G_gt, G_pred):
   """
   Approximate Matching Score
   """
   gt_edges = set(G_gt.edges())
   pred_edges = set(G_pred.edges())


   if len(gt_edges) == 0:
       return 0


   return len(gt_edges.intersection(pred_edges)) / len(gt_edges)