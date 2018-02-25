#include <stdio.h>
#include <stdlib.h>
#include "tensorflow/c/c_api.h"

#include "tf_utils.h"

#define INPUT_LAYER_NAME	"input"
#define OUTPUT_LAYER_NAME	"InceptionV3/Predictions/Reshape_1"

void print_dims (const int num_dims, const int64_t *dims)
{
  int i = 0;

  fprintf (stdout, "%i [", i, num_dims);

  for (i=0; i<num_dims; i++)
  {
    fprintf (stdout, "%i", dims[i]);

    if (i<num_dims-1)
      fprintf (stdout,", ");
  }

  fprintf (stdout, "]");
}

void get_inputs (TF_Graph *graph, TF_Operation *op)
{
  int i = 0;
  int num_inputs = 0;

  num_inputs = TF_OperationNumInputs (op);

  for (i=0; i<num_inputs; i++)
  {
    int num_dims = 0;
    int64_t *dims = NULL;
    TF_DataType type;
    TF_Input input = {op, i};

    /* tensor id */
    fprintf (stdout, "    I[%02i] ", i);

    /* type */
    type = TF_OperationInputType (input);
    fprintf (stdout, "type : %i\0", type);
    fprintf (stdout, "\n\0");
  }
}

void get_outputs (TF_Graph *graph, TF_Operation *op)
{
  int i = 0;
  int num_outputs = 0;
  TF_Status *status = NULL;

  num_outputs = TF_OperationNumOutputs (op);

  status = TF_NewStatus ();

  for (i=0; i<num_outputs; i++)
  {
    int num_dims = 0;
    int64_t *dims = NULL;
    TF_DataType type;
    TF_Output output = {op, i};

    /* id */
    fprintf (stdout, "    T[%02i] ", i);
    type = TF_OperationOutputType (output);

    fprintf (stdout, "type : %i, \0", type);

    num_dims = TF_GraphGetTensorNumDims (
      graph,
      output,
      status);

    dims = (int64_t*) malloc (num_dims * sizeof (int64_t));

    TF_GraphGetTensorShape (
      graph,
      output,
      dims,
      num_dims,
      status);

    print_dims (num_dims, dims);
    fprintf (stdout, "\n\0");

    free (dims);
  }

  /* clean up */
  TF_DeleteStatus (status);
}

void get_tensor_info (TF_Graph *graph, const char *layer_name)
{
  TF_Operation *op = NULL;
  int num_inputs = 0;
  int num_outputs = 0;
  int i = 0;

  fprintf (stdout, "TENSOR INFO '%s'\n\0", layer_name);

  op = TF_GraphOperationByName (graph, layer_name);

  if (op == NULL)
  {
    fprintf (stdout, "ERR: Could not get '%s' from graph\n\0", layer_name);
    return;
  }

  num_inputs = TF_OperationNumInputs (op);
  num_outputs = TF_OperationNumOutputs (op);
  fprintf (stdout, "SUC: '%s' %i inputs, %i outputs\n\0", layer_name, num_inputs, num_outputs);

  fprintf (stdout, "  INPUTS:\n\0");
  get_inputs (graph, op);

  fprintf (stdout, "  OUTPUTS:\n\0");
  get_outputs (graph, op);
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

  get_tensor_info (graph, INPUT_LAYER_NAME);
  get_tensor_info (graph, OUTPUT_LAYER_NAME);

  /* cleanup */
  TF_DeleteStatus (status);
  TF_DeleteGraph (graph);

  return 0;
}
