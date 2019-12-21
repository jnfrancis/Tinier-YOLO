#include "pass_through3_layer.h"
#include "cuda.h"
#include "blas.h"

#include <stdio.h>


pass_through3_layer make_pass_through3_layer(int batch, int n, int w, int h, int c, int *input_layers, int *input_sizes)
{
    fprintf(stderr," pass_through3:      no.");

    pass_through3_layer l = {0};
    l.type = PASS_THROUGH3;
    l.batch = batch;
    l.n = n;
    l.input_layers = input_layers;
    l.input_sizes = input_sizes;
    l.h = h;
    l.w = w;
    l.c = c;
    int outputs = 0;
    fprintf(stderr,"     %d + %d", input_layers[0], input_layers[1]);
    outputs += input_sizes[0];
    outputs += input_sizes[1];
    //fprintf(stderr, "   %4d x%4d x%4d \n",  w, h, c);
    //fprintf(stderr, "\n");

    l.outputs = outputs;
    l.inputs = outputs;
    l.delta =  calloc(outputs*batch, sizeof(float));
    l.output = calloc(outputs*batch, sizeof(float));;

    l.forward = forward_pass_through3_layer;
    l.backward = backward_pass_through3_layer;
    #ifdef GPU
    l.forward_gpu = forward_pass_through3_layer_gpu;
    l.backward_gpu = backward_pass_through3_layer_gpu;

    l.delta_gpu =  cuda_make_array(l.delta, outputs*batch);
    l.output_gpu = cuda_make_array(l.output, outputs*batch);
    #endif
    return l;
}

void resize_pass_through3_layer(pass_through3_layer *l, network *net)
{
    int i;
    layer first = net->layers[l->input_layers[0]];
    l->out_w = first.out_w;
    l->out_h = first.out_h;
    l->out_c = first.out_c;
    l->outputs = first.outputs;
    l->input_sizes[0] = first.outputs;
    for(i = 1; i < l->n; ++i){
        int index = l->input_layers[i];
        layer next = net->layers[index];
        l->outputs += next.outputs;
        l->input_sizes[i] = next.outputs;
        if(next.out_w == first.out_w && next.out_h == first.out_h){
            l->out_c += next.out_c;
        }else{
            printf("%d %d, %d %d\n", next.out_w, next.out_h, first.out_w, first.out_h);
            l->out_h = l->out_w = l->out_c = 0;
        }
    }
    l->inputs = l->outputs;
    l->delta =  realloc(l->delta, l->outputs*l->batch*sizeof(float));
    l->output = realloc(l->output, l->outputs*l->batch*sizeof(float));


#ifdef GPU
    cuda_free(l->output_gpu);
    cuda_free(l->delta_gpu);
    l->output_gpu  = cuda_make_array(l->output, l->outputs*l->batch);
    l->delta_gpu   = cuda_make_array(l->delta,  l->outputs*l->batch);
#endif
    
}



void forward_pass_through3_layer(const pass_through3_layer l, network net)
{
    int i, j;
    int offset = 0;
    for(i = 0; i < l.n; ++i){
        int index = l.input_layers[i];
        float *input = net.layers[index].output;
        int input_size = l.input_sizes[i];
        for(j = 0; j < l.batch; ++j){
            copy_cpu(input_size, input + j*input_size, 1, l.output + offset + j*l.outputs, 1);
        }
        offset += input_size;
    }
}



void backward_pass_through3_layer(const pass_through3_layer l, network net)
{
    int i, j;
    int offset = 0;
    for(i = 0; i < l.n; ++i){
        int index = l.input_layers[i];
        float *delta = net.layers[index].delta;
        int input_size = l.input_sizes[i];
        for(j = 0; j < l.batch; ++j){
            axpy_cpu(input_size, 1, l.delta + offset + j*l.outputs, 1, delta + j*input_size, 1);
        }
        offset += input_size;
    }
}


#ifdef GPU
void forward_pass_through3_layer_gpu(const pass_through3_layer l, network net)
{
    int i, j;
    int offset = 0;
    for(i = 0; i < l.n; ++i){
        int index = l.input_layers[i];
        float *input = net.layers[index].output_gpu;
        int input_size = l.input_sizes[i];
        for(j = 0; j < l.batch; ++j){
            copy_gpu(input_size, input + j*input_size, 1, l.output_gpu + offset + j*l.outputs, 1);
        }
        offset += input_size;
    }
}


void backward_pass_through3_layer_gpu(const pass_through3_layer l, network net)
{
    int i, j;
    int offset = 0;
    for(i = 0; i < l.n; ++i){
        int index = l.input_layers[i];
        float *delta = net.layers[index].delta_gpu;
        int input_size = l.input_sizes[i];
        for(j = 0; j < l.batch; ++j){
            axpy_gpu(input_size, 1, l.delta_gpu + offset + j*l.outputs, 1, delta + j*input_size, 1);
        }
        offset += input_size;
    }
}
#endif
