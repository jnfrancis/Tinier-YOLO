#ifndef GET_SQUEEZE_LAYER_H
#define GET_SQUEEZE_LAYER_H
#include "network.h"
#include "layer.h"

typedef layer get_squeeze_layer;

get_squeeze_layer make_get_squeeze_layer(int batch, int n, int w, int h, int c, int *input_layers, int *input_size);
void forward_get_squeeze_layer(const get_squeeze_layer l, network net);
void backward_get_squeeze_layer(const get_squeeze_layer l, network net);
void resize_get_squeeze_layer(get_squeeze_layer *l, network *net);

#ifdef GPU
void forward_get_squeeze_layer_gpu(const get_squeeze_layer l, network net);
void backward_get_squeeze_layer_gpu(const get_squeeze_layer l, network net);
#endif

#endif
