#include <stdio.h>
#include <stdlib.h>
#include "tensorflow/c/c_api.h"

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

  fprintf (stdout, "SUC: imported graph\n\0");

  TF_DeleteStatus (status);
  TF_DeleteGraph (graph);

  return 0;
}
