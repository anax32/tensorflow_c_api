#include <stdio.h>
#include <stdlib.h>
#include "include/tensorflow/c/c_api.h"

void free_buffer (void* data, size_t length)
{
  free (data);
}

TF_Buffer* read_file (const char* file)
{
  FILE *f = NULL;
  long fsize = 0;
  void *data = NULL;
  TF_Buffer* buf = NULL;

  if ((f = fopen(file, "rb")) == NULL)
  {
    return NULL;
  }

  fseek (f, 0, SEEK_END);
  fsize = ftell (f);
  fseek (f, 0, SEEK_SET);

  if (fsize < 1)
  {
    fclose (f);
    return NULL;
  }

  data = malloc (fsize);
  fread (data, fsize, 1, f);
  fclose (f);

  buf = TF_NewBuffer ();
  buf->data = data;
  buf->length = fsize;
  buf->data_deallocator = free_buffer;
  return buf;
}

void enumerate_operation_inputs (TF_Graph* graph, TF_Operation* op)
{
  int num_inputs = 0;

  num_inputs = TF_OperationNumInputs (op);

  fprintf (stdout, "  inputs:\n\0");

  for (int i=0; i<num_inputs; i++)
  {
    TF_Input input = {op, i};

    fprintf (stdout, "    %03i : ", i);

    TF_DataType type = TF_OperationInputType (input);
    fprintf (stdout, "type : %i", type);

    fprintf (stdout, "\n\0");
  }
}

void enumerate_operation_outputs (TF_Graph* graph, TF_Operation* op)
{
  TF_Status* status = NULL;
  int num_outputs = 0;

  num_outputs = TF_OperationNumOutputs (op);

  status = TF_NewStatus ();

  fprintf (stdout, "  outputs:\n\0");

  for (int i=0; i<num_outputs; i++)
  {
    TF_Output output = {op, i};
    TF_DataType type = 0;
    int num_dims = 0;
    int64_t * dims = NULL;

    fprintf (stdout, "    %03i : ", i);

    type = TF_OperationOutputType (output);
    fprintf (stdout, "type : %i, \0", i, type);

    num_dims = TF_GraphGetTensorNumDims (graph, output, status);

    if (TF_GetCode (status) != TF_OK)
    {
      fprintf (stdout, "ERR: could not get tensor dimensionality\0");
      continue;
    }

    dims = (int64_t*) malloc (num_dims * sizeof (int64_t));
    TF_GraphGetTensorShape (graph, output, dims, num_dims, status);

    if (TF_GetCode (status) != TF_OK)
    {
      fprintf (stdout, "ERR: could not get tensor shape\n\0");
      free (dims);
      continue;
    }

    fprintf (stdout, " % 3i [", num_dims);
    for (int j=0;j<num_dims;j++)
    {
      fprintf (stdout, "% 3i", dims[j]);

      if (j<num_dims - 1)
      {
        fprintf (stdout, ", ");
      }
    }
    fprintf (stdout, "] ");
    free (dims);

    fprintf (stdout, "\n\0");
  }

  TF_DeleteStatus (status);
}

void enumerate_operations (TF_Graph* graph)
{
  TF_Operation* op = NULL;
  size_t pos = 0;

  while ((op = TF_GraphNextOperation (graph, &pos)) != NULL)
  {
    const char *name = TF_OperationName (op);
    const char *type = TF_OperationOpType (op);
    const char *device = TF_OperationDevice (op);

    int num_outputs = TF_OperationNumOutputs (op);
/*    TF_DataType output_type = TF_OperationOutputType (op);*/
/*    int output_list_length = TF_OperationOutputListLength (op, arg_name, status);*/

    int num_inputs = TF_OperationNumInputs (op);
/*    TF_DataType input_type = TF_OperationInputType (op);*/

    fprintf (stdout, "%03i: %s [%s, %s] (%i) -> (%i)\n\0", pos, name, type, device, num_inputs, num_outputs);

    enumerate_operation_inputs (graph, op);
    enumerate_operation_outputs (graph, op);
  }
}

void enumerate_functions (TF_Graph* graph)
{
  int num_functions = 0;

/*  num_functions = TF_GraphNumFunctions (graph); */

  fprintf (stdout, "number of functions : %03i\n\0", num_functions);
}

int main (int argc, char **argv)
{
  TF_Buffer* graph_def = NULL;
  TF_Graph* graph = NULL;
  TF_Status* status = NULL;
  TF_ImportGraphDefOptions* opts = NULL;

  if ((graph_def = read_file (argv[1])) == NULL)
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

  fprintf (stdout, "enumerating graph operations:\n\0");
  enumerate_operations (graph);

  TF_DeleteStatus (status);
  TF_DeleteGraph (graph);

  return 0;
}
