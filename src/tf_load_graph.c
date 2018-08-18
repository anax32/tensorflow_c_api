#include <stdio.h>
#include <stdlib.h>
#include "tensorflow/c/c_api.h"

#include "tf_utils.h"

int main (int argc, char **argv)
{
  TF_Buffer* graph_def = NULL;
  TF_Graph* graph = NULL;
  TF_Status* status = NULL;
  TF_ImportGraphDefOptions* opts = NULL;

  if ((graph_def = buffer_read_from_file (argv[1])) == NULL)
  {
    return 1;
  }

  graph = TF_NewGraph ();
  status = TF_NewStatus ();
  opts = TF_NewImportGraphDefOptions ();

  TF_GraphImportGraphDef (graph, graph_def, opts, status);
  TF_DeleteImportGraphDefOptions (opts);
  TF_DeleteBuffer (graph_def);

  /* check the status */
  if (TF_GetCode (status) != TF_OK)
  {
    fprintf (stderr, "ERR: Unable to import graph '%s'\n\0", TF_Message (status));

    TF_DeleteStatus (status);
    TF_DeleteGraph (graph);

    return 1;
  }

  fprintf (stdout, "SUC: imported graph\n\0");

  TF_DeleteStatus (status);
  TF_DeleteGraph (graph);

  return 0;
}
