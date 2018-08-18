#include <stdio.h>
#include <stdlib.h>

void buffer_deallocate_callback (void* data, size_t length)
{
  free (data);
}

TF_Buffer* buffer_read_from_file (const char* file)
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
  buf->data_deallocator = buffer_deallocate_callback;
  return buf;
}
