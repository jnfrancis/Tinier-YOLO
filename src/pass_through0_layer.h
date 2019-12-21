#ifndef PASS_THROUGH0_LAYER_H
#define PASS_THROUGH0_LAYER_H
#include "network.h"
#include "layer.h"

typedef layer pass_through0_layer;

pass_through0_layer make_pass_through0_layer(int batch, int n, int w, int h, int c, int *input_layers, int *input_size);
void forward_pass_through0_layer(const pass_through0_layer l, network net);
void backward_pass_through0_layer(const pass_through0_layer l, network net);
void resize_pass_through0_layer(pass_through0_layer *l, network *net);

#ifdef GPU
void forward_pass_through0_layer_gpu(const pass_through0_layer l, network net);
void backward_pass_through0_layer_gpu(const pass_through0_layer l, network net);
#endif

#endif
