#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <png.h>
#include "tensorflow/c/c_api.h"

#include "tf_utils.h"

#define INPUT_LAYER_NAME	"input"
#define OUTPUT_LAYER_NAME	"InceptionV3/Predictions/Reshape_1"

/**
 * callback to deallocate the memory for a tensor
 * FIXME: this is NOT called for the output tensor, no idea why
 */
void deallocate_tensor (void *data, size_t len, void *arg)
{
/*  char *label = (char *)arg;
  fprintf (stdout, "freeing tensor at 0x%x for %i bytes ('%s')\n\0", data, len, label);*/
  free (data);
}

/**
 * get the location of the largest element in the tensor
 * used to get the high value from the softmax output layer
 * only useful when the shape is [1,N,1,1] or something
 */
size_t argmax (TF_Tensor* t)
{
  const int num_dims = TF_NumDims (t);
  size_t element_count = 0;
  int i = 0;
  const float* data = TF_TensorData (t);
  float mx = 0.0f;
  size_t mx_i = 0;

  for (i=0, element_count=1; i<num_dims; i++)
  {
    element_count *= TF_Dim (t, i);
  }

  for (i=1, mx=data[0]; i<element_count; i++)
  {
    if (data[i] > mx)
    {
      mx = data[i];
      mx_i = i;
    }
  }

  return mx_i;
}

/**
 * simple tuple wrapper
 */
typedef struct image_s
{
  float *data;
  size_t element_count;
} image_t;

/**
 * load a png file from disk
 * the png must be byte, three channel and 299*299 size
 */
image_t* load_png (const char *filename)
{
  FILE *f = NULL;
  png_structp hdr;
  png_infop info;
  int w, h;
  png_byte ct;
  png_byte bd;
  unsigned char *byte_buf = NULL;
  unsigned char **row_pointers = NULL;
  float *float_buf = NULL;

  /* read the png image shape attributes */
  if ((f = fopen (filename, "rb")) == NULL)
  {
    fprintf (stderr, "ERR: Could not read '%s'\n\0", filename);
    return NULL;
  }

  hdr = png_create_read_struct (PNG_LIBPNG_VER_STRING, NULL, NULL, NULL);
  info = png_create_info_struct (hdr);
  png_init_io (hdr, f);
  png_read_info (hdr, info);
  w = png_get_image_width (hdr, info);
  h = png_get_image_height (hdr, info);
  ct = png_get_color_type (hdr, info);
  bd = png_get_bit_depth (hdr, info);
  png_destroy_read_struct (&hdr, &info, NULL);
  fclose (f);

  /* check the attributes are within our scope */
  if (ct != PNG_COLOR_TYPE_RGB)
  {
    fprintf (stderr, "ERR: '%s' is not RGB\n\0", filename);
    return NULL;
  }

  if ((w != 299) || (h != 299))
  {
    fprintf (stderr, "ERR: '%s' dimensions are not inception dimensions (%i, %i) != (299, 299)\n\0", filename, w, h);
    return NULL;
  }

  if (bd != 8)
  {
    fprintf (stderr, "ERR: cannot read '%s' with bit depth %i\n\0", filename, bd);
    return NULL;
  }

  /* allocate storage for the image data
     and row pointers */
  byte_buf = (unsigned char *) malloc (w*h*3*sizeof (unsigned char));
  row_pointers = (unsigned char **) malloc (h*sizeof (unsigned char *));

  for (int i=0; i<h; i++)
  {
    row_pointers[i] = &byte_buf[i*w*3];
  }

  /* read the image data */
  f = fopen (filename, "rb");
  hdr = png_create_read_struct (PNG_LIBPNG_VER_STRING, NULL, NULL, NULL);
  info = png_create_info_struct (hdr);
  png_init_io (hdr, f);
  png_read_info (hdr, info);
  png_read_image (hdr, row_pointers);
  png_read_end (hdr, NULL);
  png_destroy_read_struct (&hdr, &info, NULL);
  fclose (f);

  /* convert to float */
  float_buf = (float *) malloc (w*h*3*sizeof (float));

  for (int i=0; i<w*h*3; i++)
  {
    float_buf[i] = (float)byte_buf[i];
  }

  free (byte_buf);
  free (row_pointers);

  image_t* img = (image_t*) malloc (sizeof (image_t));
  img->data = float_buf;
  img->element_count = w*h*3;
  return img;
}

/**
 * inception_v3 requires images are in the range [-1, 1]
 */
void preprocess_inception (image_t* image)
{
  int i;

  for (i=0; i<image->element_count; i++)
  {
    image->data[i] /= 255.0f;
    image->data[i] -= 0.5f;
    image->data[i] *= 2.0f;
  }
}

/**
 * run a graph in a session
 * given input and output tensors should be allocated
 * to the correct shapes already
 */
int run_session (TF_Graph* graph,
                 TF_Output* input, TF_Tensor** input_tensor,
                 TF_Output* output, TF_Tensor** output_tensor)
{
  TF_Status* status = NULL;
  TF_Session* sess = NULL;
  TF_SessionOptions* options = NULL;
  int retval = 0;

  /* create a session */
  status = TF_NewStatus ();
  options = TF_NewSessionOptions ();

  sess = TF_NewSession (graph, options, status);

  if (TF_GetCode (status) != TF_OK)
  {
    fprintf (stderr, "ERR: Could not get session: '%s'\n\0", TF_Message (status));
    TF_DeleteSessionOptions (options);
    TF_DeleteStatus (status);
    return 1;
  }

  TF_SessionRun (sess,
    NULL, /* run options */
    input, input_tensor, 1, /* input tensors, input tensor values, number of inputs */
    output, output_tensor, 1, /* output tensors, output tensor values, number of outputs */
    NULL, 0, /* target operations, number of targets */
    NULL, /* run metadata */
    status);

  if (TF_GetCode (status) != TF_OK)
  {
    fprintf (stderr, "ERR: Could not run session: '%s'\n\0", TF_Message (status));
    retval = 2;
  }
  else
  {
    retval = 0;
  }

  /* cleanup */
  TF_CloseSession (sess, status);

  if (TF_GetCode (status) != TF_OK)
  {
    fprintf (stderr, "ERR: Could not close session: '%s'\n\0", TF_Message (status));
  }

  TF_DeleteSession (sess, status);

  if (TF_GetCode (status) != TF_OK)
  {
    fprintf (stderr, "ERR: Could not delete session: '%s'\n\0", TF_Message (status));
  }

  return retval;
}

/**
 * setup the data context for the graph;
 * create the input and output tensors
 * pass to a session
 * get the results from the output tensor
 */
int establish_context (TF_Graph* graph, const char *input_layer_name, const char *output_layer_name, const image_t* input_data)
{
  TF_Status* status = NULL;
  TF_Operation* input_op = NULL;
  TF_Operation* output_op = NULL;
  TF_Output input; /* output from the input layer, i.e., the data */
  TF_Output output; /* output from the output layer, i.e., the prediction */
  int input_num_dims = 0;
  int output_num_dims = 0;
  int64_t* input_dims = NULL;
  int64_t* output_dims = NULL;
  size_t input_element_count = 1;
  size_t output_element_count = 1;
  TF_Tensor* input_tensor = NULL;
  TF_Tensor* output_tensor = NULL;
  float* input_tensor_data = NULL;
  float* output_tensor_data = NULL;
  int i = 0;

  /* get the input and output layers */
  if ((input_op = TF_GraphOperationByName (graph, input_layer_name)) == NULL)
  {
    fprintf (stderr, "ERR: Could not get input operation '%s' from graph\n\0", input_layer_name);
    return 1;
  }

  if ((output_op = TF_GraphOperationByName (graph, output_layer_name)) == NULL)
  {
    fprintf (stderr, "ERR: Could not get output operation '%s' from graph\n\0", output_layer_name);
    return 2;
  }

  input.oper = input_op;
  input.index = 0;
  output.oper = output_op;
  output.index = 0;

  /* allocate the tensors for input and output */
  input_num_dims = TF_GraphGetTensorNumDims (
      graph,
      input,
      status);

  input_dims = (int64_t*) malloc (input_num_dims * sizeof (int64_t));

  TF_GraphGetTensorShape (
    graph,
    input,
    input_dims,
    input_num_dims,
    status);

  for (i=0; i<input_num_dims; i++)
  {
    input_element_count *= input_dims[i];
  }

  output_num_dims = TF_GraphGetTensorNumDims (
      graph,
      output,
      status);

  output_dims = (int64_t*) malloc (output_num_dims * sizeof (int64_t));

  TF_GraphGetTensorShape (
    graph,
    output,
    output_dims,
    output_num_dims,
    status);

  for (i=0; i<output_num_dims; i++)
  {
    output_element_count *= output_dims[i];
  }

  /* allocate the tensor storage */
  input_tensor_data = input_data->data;
  output_tensor_data = (float *) malloc (output_element_count * sizeof (float));
  memset (output_tensor_data, 0, output_element_count * sizeof (float));

  input_tensor = TF_NewTensor (
    TF_FLOAT,
    input_dims,
    input_num_dims,
    input_tensor_data,
    input_element_count * sizeof (float),
    deallocate_tensor,
    (void*)"input");

  /* memory leak here: deallocate_tensor is not called */
  output_tensor = TF_NewTensor (
    TF_FLOAT,
    output_dims,
    output_num_dims,
    output_tensor_data,
    output_element_count * sizeof (float),
    deallocate_tensor,
    (void*)"output");

  /* execute the graph */
  run_session (
    graph,
    &input,
    &input_tensor,
    &output,
    &output_tensor);

  /* get the results */
  int idx = argmax (output_tensor);
  fprintf (stdout, "%i\n\0", idx);

  /* cleanup */
  free (input_dims);
  free (output_dims);
  TF_DeleteTensor (input_tensor);
  TF_DeleteTensor (output_tensor); /* does not invoke deallocate_tensor */
  return 0;
}

int main (int argc, char **argv)
{
  TF_Buffer* graph_def = NULL;
  TF_Graph* graph = NULL;
  TF_Status* status = NULL;
  TF_ImportGraphDefOptions* opts = NULL;
  image_t* image = NULL;

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

  /* read the image to be classified */
  if ((image = load_png (argv[2])) == NULL)
  {
    fprintf (stderr, "ERR: unable to read image '%s'\n\0", argv[2]);
    TF_DeleteStatus (status);
    TF_DeleteGraph (graph);
    return 1;
  }

  preprocess_inception (image);

  /* classify the image */
//  run_session (
  establish_context (
    graph,
    INPUT_LAYER_NAME,
    OUTPUT_LAYER_NAME,
    image);

  /* cleanup */
  free (image); /* image->data should be free'd by deallocate_tensor () */
  TF_DeleteStatus (status);
  TF_DeleteGraph (graph);

  return 0;
}
