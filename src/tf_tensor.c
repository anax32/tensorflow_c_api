#include <stdio.h>
#include <stdlib.h>
#include "tensorflow/c/c_api.h"

#include "tf_utils.h"

void tensor_deallocation (void *data, size_t len, void * arg)
{
  fprintf (stdout, "freeing tensor at 0x%x for %i bytes\n\0", data, len);
  free (data);
}

int main (int argc, char **argv)
{
  TF_Tensor *t = NULL;
  int64_t  dims[] = {16,16,16,16};
  int num_dims = sizeof (dims)/sizeof(dims[0]);
  int i = 0;
  int element_count = 1;
  size_t tf_element_size = 0;
  float *initial_data = NULL;
  void *tf_data = NULL;
  size_t match_count = 0;

  /* get the number of elements */
  for (i=0; i<num_dims; i++)
  {
    element_count *= dims[i];
  }

  /* display some info about the tensor we want */
  fprintf (stdout, "dims : %i, [", num_dims);
  for (i=0; i<num_dims; i++)
  {
    fprintf (stdout, "%i, ", dims[i]);
  }
  fprintf (stdout, "]\n\0");

  /* allocate the data */
  initial_data = (float *) malloc (element_count * sizeof (float));

  for (i=0; i<element_count; i++)
  {
    initial_data[i] = (float)i;
  }

  fprintf (stdout, "allocated at 0x%x %i bytes\n\0", initial_data, element_count*sizeof (float));

  /* create the tensor */
  t = TF_NewTensor (
    TF_FLOAT,
    dims,
    num_dims,
    initial_data,
    element_count * sizeof (float),
    tensor_deallocation,
    NULL);

  /* check what we created matches what we requested */
  if (TF_TensorType (t) != TF_FLOAT)
  {
    fprintf (stdout, "ERR: wrong tensor type, %i != %i\n\0", TF_TensorType (t), TF_FLOAT);
  }

  if (TF_NumDims (t) != num_dims)
  {
    fprintf (stdout, "ERR: wrong number of dimensions, %i != %i\n\0", TF_NumDims (t), num_dims);
  }

  for (i=0; i<num_dims; i++)
  {
    if (TF_Dim (t, i) != dims[i])
    {
      fprintf (stdout, "ERR: wrong dimension size for dim %i, %i != %i\n\0", i, TF_Dim (t, i), dims[i]);
    }
  }

  if (TF_TensorByteSize (t) != element_count * sizeof (float))
  {
    fprintf (stdout, "ERR: wrong tensor byte size, %i != %i\n\0", TF_TensorByteSize (t), element_count * sizeof (float));
    fprintf (stdout, "     host float size : %i\n\0", sizeof (float));
    fprintf (stdout, "     tf float size : %i\n\0", tf_element_size);
  }

  tf_data = TF_TensorData (t);
  tf_element_size = TF_DataTypeSize (TF_FLOAT);

  fprintf (stdout, "SUC: got tensor pointer 0x%x with element size %i\n\0", tf_data, tf_element_size);

  for (i=0; i<element_count; i++)
  {
    float tf_element = *(float*)(tf_data+(tf_element_size*i));

    if (tf_element != initial_data[i])
    {
      match_count++;
/*      fprintf (stdout, "ERR: element %i does not match, %f != %f\n\0", i, tf_element, initial_data[i]);
*/
    }
  }

  if (match_count > 0)
  {
    fprintf (stdout, "ERR: %i elements did not match\n\0", match_count);
  }

  fprintf (stdout, "SUC: tensor compared\n\0");

  TF_DeleteTensor (t);

  return 0;
}
