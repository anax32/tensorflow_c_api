#include <stdio.h>
#include <stdlib.h>
#include "tensorflow/c/c_api.h"

#include "tf_utils.h"

int run_session (TF_Graph* graph)
{
  TF_Status* status = NULL;
  TF_Session* sess = NULL;
  TF_SessionOptions* options = NULL;

  status = TF_NewStatus ();
  options = TF_NewSessionOptions ();

  sess = TF_NewSession (graph, options, status);

  if (TF_GetCode (status) != TF_OK)
  {
    fprintf (stdout, "ERR: Could not get session: '%s'\n\0", TF_Message (status));
    TF_DeleteSessionOptions (options);
    TF_DeleteStatus (status);
    return 1;
  }

  TF_SessionRun (sess,
    NULL, /* run options */
    NULL, NULL, 0, /* input tensors, input tensor values, number of inputs */
    NULL, NULL, 0, /* output tensors, output tensor values, number of outputs */
    NULL, 0, /* target operations, number of targets */
    NULL, /* run metadata */
    status);

  if (TF_GetCode (status) != TF_OK)
  {
    fprintf (stdout, "ERR: Could not run session: '%s'\n\0", TF_Message (status));
    TF_DeleteSessionOptions (options);
    TF_DeleteStatus (status);
    return 2;
  }

  TF_CloseSession (sess, status);

  if (TF_GetCode (status) != TF_OK)
  {
    fprintf (stdout, "ERR: Could not close session: '%s'\n\0", TF_Message (status));
    TF_DeleteSessionOptions (options);
    TF_DeleteStatus (status);
    return 3;
  }

  TF_DeleteSession (sess, status);

  if (TF_GetCode (status) != TF_OK)
  {
    fprintf (stdout, "ERR: Could not delete session: '%s'\n\0", TF_Message (status));
    TF_DeleteSessionOptions (options);
    TF_DeleteStatus (status);
    return 4;
  }

  TF_DeleteSessionOptions (options);
  TF_DeleteStatus (status);
  return 0;
}

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

  run_session (graph);

  TF_DeleteStatus (status);
  TF_DeleteGraph (graph);

  return 0;
}
