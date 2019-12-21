#ifndef MERGE_EXPAND_LAYER_H
#define MERGE_EXPAND_LAYER_H
#include "network.h"
#include "layer.h"

typedef layer merge_expand_layer;

merge_expand_layer make_merge_expand_layer(int batch, int n, int w, int h, int c,  int *input_layers, int *input_size);
void forward_merge_expand_layer(const merge_expand_layer l, network net);
void backward_merge_expand_layer(const merge_expand_layer l, network net);
void resize_merge_expand_layer(merge_expand_layer *l, network *net);

#ifdef GPU
void forward_merge_expand_layer_gpu(const merge_expand_layer l, network net);
void backward_merge_expand_layer_gpu(const merge_expand_layer l, network net);
#endif

#endif
