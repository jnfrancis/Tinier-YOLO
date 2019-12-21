#ifndef DENSE_LAYER_H
#define DENSE_LAYER_H
#include "network.h"
#include "layer.h"

typedef layer dense_layer;

dense_layer make_dense_layer(int batch, int n, int w, int h, int c,  int *input_layers, int *input_size);
void forward_dense_layer(const dense_layer l, network net);
void backward_dense_layer(const dense_layer l, network net);
void resize_dense_layer(dense_layer *l, network *net);

#ifdef GPU
void forward_dense_layer_gpu(const dense_layer l, network net);
void backward_dense_layer_gpu(const dense_layer l, network net);
#endif

#endif
