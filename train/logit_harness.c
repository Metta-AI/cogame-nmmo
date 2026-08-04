/* Parity harness: run the puffernet MMONet forward on deterministic
 * obs and dump the raw decoder outputs (26 logits + value).
 *
 * The MMONet construction and forward are copied VERBATIM from
 * vendor/upstream/nmmo3.c (demo section) minus the sampling call, so
 * the dumped values are the pre-softmax network outputs the league
 * brain computes.
 *
 * Build: clang -O2 -I vendor/upstream train/logit_harness.c -o train/logit_harness -lm
 * Run:   ./train/logit_harness <weights.bin> <num_agents> <steps>
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "puffernet.h"

typedef struct MMONet MMONet;
struct MMONet {
    int num_agents;
    float* ob_map;
    int* ob_player_discrete;
    float* ob_player_continuous;
    float* ob_reward;
    Conv2D* map_conv1;
    ReLU* map_relu;
    Conv2D* map_conv2;
    Embedding* player_embed;
    float* proj_buffer;
    Affine* proj;
    ReLU* proj_relu;
    Linear* decoder;
    MinGRU* mingru;
    Multidiscrete* multidiscrete;
};

MMONet* init_mmonet(Weights* weights, int num_agents) {
    MMONet* net = calloc(1, sizeof(MMONet));
    int hidden = 512;
    net->num_agents = num_agents;
    net->ob_map = calloc(num_agents*11*15*59, sizeof(float));
    net->ob_player_discrete = calloc(num_agents*47, sizeof(int));
    net->ob_player_continuous = calloc(num_agents*47, sizeof(float));
    net->ob_reward = calloc(num_agents*10, sizeof(float));
    net->map_conv1 = make_conv2d(weights, num_agents, 15, 11, 59, 128, 5, 3);
    net->map_relu = make_relu(num_agents, 128*3*4);
    net->map_conv2 = make_conv2d(weights, num_agents, 4, 3, 128, 128, 3, 1);
    net->player_embed = make_embedding(weights, num_agents*47, 128, 32);
    net->proj_buffer = calloc(num_agents*1817, sizeof(float));
    net->proj = make_affine(weights, num_agents, 1817, hidden);
    net->proj_relu = make_relu(num_agents, hidden);
    net->decoder = make_linear(weights, num_agents, hidden, 26 + 1);
    net->mingru = make_mingru(weights, num_agents, hidden, 4);
    int logit_sizes[1] = {26};
    net->multidiscrete = make_multidiscrete(num_agents, logit_sizes, 1);
    return net;
}

void forward_logits(MMONet* net, unsigned char* observations) {
    memset(net->ob_map, 0, net->num_agents*11*15*59*sizeof(float));
    int factors[10] = {4, 4, 17, 5, 3, 5, 5, 5, 7, 4};
    float (*ob_map)[59][11][15] = (float (*)[59][11][15])net->ob_map;
    for (int b = 0; b < net->num_agents; b++) {
        int b_offset = b*(11*15*10 + 47 + 10);
        for (int i = 0; i < 11; i++) {
            for (int j = 0; j < 15; j++) {
                int f_offset = 0;
                for (int f = 0; f < 10; f++) {
                    int obs_idx = f_offset + observations[b_offset + i*15*10 + j*10 + f];
                    ob_map[b][obs_idx][i][j] = 1;
                    f_offset += factors[f];
                }
            }
        }
    }
    conv2d(net->map_conv1, net->ob_map);
    relu(net->map_relu, net->map_conv1->output);
    conv2d(net->map_conv2, net->map_relu->output);

    for (int b = 0; b < net->num_agents; b++) {
        for (int i = 0; i < 47; i++) {
            unsigned char ob = observations[b*(11*15*10 + 47 + 10) + 11*15*10 + i];
            net->ob_player_discrete[b*47 + i] = ob;
            net->ob_player_continuous[b*47 + i] = ob;
        }
    }
    embedding(net->player_embed, net->ob_player_discrete);

    for (int b = 0; b < net->num_agents; b++) {
        for (int i = 0; i < 10; i++) {
            net->ob_reward[b*10 + i] = observations[b*(11*15*10 + 47 + 10) + 11*15*10 + 47 + i];
        }
    }

    for (int b = 0; b < net->num_agents; b++) {
        int b_offset = b*1817;
        for (int i = 0; i < 256; i++)
            net->proj_buffer[b_offset + i] = net->map_conv2->output[b*256 + i];
        b_offset += 256;
        for (int i = 0; i < 47*32; i++)
            net->proj_buffer[b_offset + i] = net->player_embed->output[b*47*32 + i];
        b_offset += 47*32;
        for (int i = 0; i < 47; i++)
            net->proj_buffer[b_offset + i] = net->ob_player_continuous[b*47 + i];
        b_offset += 47;
        for (int i = 0; i < 10; i++)
            net->proj_buffer[b_offset + i] = net->ob_reward[b*10 + i];
    }

    affine(net->proj, net->proj_buffer);
    relu(net->proj_relu, net->proj->output);
    mingru(net->mingru, net->proj_relu->output);
    linear(net->decoder, net->mingru->output);
}

int main(int argc, char** argv) {
    if (argc < 4) {
        fprintf(stderr, "usage: %s weights.bin num_agents steps\n", argv[0]);
        return 1;
    }
    Weights* weights = load_weights(argv[1]);
    int num_agents = atoi(argv[2]);
    int steps = atoi(argv[3]);
    MMONet* net = init_mmonet(weights, num_agents);
    int obs_size = 11*15*10 + 47 + 10;
    unsigned char* obs = calloc(num_agents*obs_size, 1);
    for (int t = 0; t < steps; t++) {
        /* deterministic, step-varying obs; player bytes stay < 128 */
        for (int i = 0; i < num_agents*obs_size; i++)
            obs[i] = (unsigned char)((i + 7*t) % 3);
        forward_logits(net, obs);
        for (int b = 0; b < num_agents; b++) {
            for (int i = 0; i < 27; i++)
                printf("%.9g ", net->decoder->output[b*27 + i]);
            printf("\n");
        }
    }
    return 0;
}
