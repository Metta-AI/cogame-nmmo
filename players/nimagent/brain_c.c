/* Baseline-brain C core for the Nim bench: puffernet MMONet forward
 * producing raw logits (26 + value). Construction/forward copied
 * verbatim from vendor/upstream/nmmo3.c (same code as
 * train/logit_harness.c, whose outputs match the torch mirror to
 * 1.4e-4 and the league wasm brain by construction). Sampling is done
 * caller-side (Nim) from the logits.
 */
#include <stdlib.h>
#include <string.h>
#include "puffernet.h"

typedef struct BrainNet BrainNet;
struct BrainNet {
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
};

void* brain_create(const char* weights_path, int num_agents) {
    Weights* weights = load_weights(weights_path);
    if (!weights) return 0;
    BrainNet* net = calloc(1, sizeof(BrainNet));
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
    return net;
}

/* obs: num_agents*1707 bytes; terminals: num_agents floats (1.0 resets
 * that agent's recurrent state, cleared here); out: num_agents*27. */
void brain_forward(void* handle, unsigned char* observations,
                   float* terminals, float* out) {
    BrainNet* net = (BrainNet*)handle;
    for (int b = 0; b < net->num_agents; b++) {
        if (terminals[b] > 0.5f) {
            for (int l = 0; l < net->mingru->num_layers; l++) {
                memset(net->mingru->state
                    + l * net->mingru->batch_size * net->mingru->hidden_size
                    + b * net->mingru->hidden_size,
                    0, net->mingru->hidden_size * sizeof(float));
            }
            terminals[b] = 0.0f;
        }
    }
    memset(net->ob_map, 0, net->num_agents*11*15*59*sizeof(float));
    int factors[10] = {4, 4, 17, 5, 3, 5, 5, 5, 7, 4};
    float (*ob_map)[59][11][15] = (float (*)[59][11][15])net->ob_map;
    for (int b = 0; b < net->num_agents; b++) {
        int b_offset = b*(11*15*10 + 47 + 10);
        for (int i = 0; i < 11; i++) {
            for (int j = 0; j < 15; j++) {
                int f_offset = 0;
                for (int f = 0; f < 10; f++) {
                    int obs_idx = f_offset
                        + observations[b_offset + i*15*10 + j*10 + f];
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
            unsigned char ob =
                observations[b*(11*15*10 + 47 + 10) + 11*15*10 + i];
            net->ob_player_discrete[b*47 + i] = ob;
            net->ob_player_continuous[b*47 + i] = ob;
        }
    }
    embedding(net->player_embed, net->ob_player_discrete);

    for (int b = 0; b < net->num_agents; b++) {
        for (int i = 0; i < 10; i++) {
            net->ob_reward[b*10 + i] =
                observations[b*(11*15*10 + 47 + 10) + 11*15*10 + 47 + i];
        }
    }

    for (int b = 0; b < net->num_agents; b++) {
        int b_offset = b*1817;
        for (int i = 0; i < 256; i++)
            net->proj_buffer[b_offset + i] =
                net->map_conv2->output[b*256 + i];
        b_offset += 256;
        for (int i = 0; i < 47*32; i++)
            net->proj_buffer[b_offset + i] =
                net->player_embed->output[b*47*32 + i];
        b_offset += 47*32;
        for (int i = 0; i < 47; i++)
            net->proj_buffer[b_offset + i] =
                net->ob_player_continuous[b*47 + i];
        b_offset += 47;
        for (int i = 0; i < 10; i++)
            net->proj_buffer[b_offset + i] = net->ob_reward[b*10 + i];
    }

    affine(net->proj, net->proj_buffer);
    relu(net->proj_relu, net->proj->output);
    mingru(net->mingru, net->proj_relu->output);
    linear(net->decoder, net->mingru->output);
    memcpy(out, net->decoder->output,
           net->num_agents * 27 * sizeof(float));
}
