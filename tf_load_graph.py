import sys
import tensorflow as tf

def load_graph (model_file):
  graph = tf.Graph ()
  graph_def = tf.GraphDef ()

  with open (model_file, "rb") as f:
    graph_def.ParseFromString (f.read ())

  with graph.as_default ():
    tf.import_graph_def (graph_def)

  return graph

if __name__ == "__main__":
  graph = load_graph (sys.argv[1])

  print (graph)
