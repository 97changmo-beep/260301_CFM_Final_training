/*#########################################################################
# Mass Spec Prediction and Identification of Metabolites
#
# nn_param.cpp
#
# Description: 	Class for parameterization of fragmentation probabilities
#				within the bayesian network fragmentation trees.
#
# Copyright (c) 2013, Felicity Allen
# Copyright (c) 2016, Fei Wang
# All rights reserved.

# This file is part of the cfm-id project.
# The contents are covered by the terms of the GNU Lesser General Public
# License, which is included in the file license.txt, found at the root
# of the cfm source tree.
#########################################################################*/

#include "NNParam.h"
#include "Config.h"

#ifdef _OPENMP
#include <omp.h>
#endif

NNParam::NNParam(std::vector<std::string> a_feature_list, int a_num_energy_levels,
                 std::vector<int> &a_hlayer_num_nodes, std::vector<int> &a_act_func_ids,
                 std::vector<float> &a_dropout_probs, boost::container::vector<bool> & a_is_frozen):
        h_layer_num_nodes(a_hlayer_num_nodes), act_func_ids(a_act_func_ids),
        hlayer_dropout_probs(a_dropout_probs), hlayer_is_frozen(a_is_frozen),
        Param(a_feature_list, a_num_energy_levels) {

    //The weight length was set in the Param constructor, now alter it to match the neural net layout
    unsigned int num_features = weights.size() / num_energy_levels;
    input_layer_node_num = num_features;

    // set input layer
    int total_len = 0, num_input;
    auto hlayer_node_num = h_layer_num_nodes.begin();
    int num_input_layer_weights = (*hlayer_node_num) * num_features;
    total_len += num_input_layer_weights;//The first layer takes the fv as input (which has in-built bias).
    num_weights_per_layer.push_back(num_input_layer_weights);

    // handle hidden layers
    total_nodes = (*hlayer_node_num);
    num_input = (*hlayer_node_num);
    ++hlayer_node_num;

    for (; hlayer_node_num != h_layer_num_nodes.end(); ++hlayer_node_num) {
        int num_weights = (*hlayer_node_num) * (num_input + 1);
        num_weights_per_layer.push_back(num_weights);

        total_len += num_weights;
        total_nodes += *hlayer_node_num;
        num_input = *hlayer_node_num;
    }
    weights.resize(total_len * num_energy_levels);

    //Set pointers to the activation functions and their derivatives used in each layer
    setActivationFunctionsFromIds();

    //Resize some temporary a and z vectors so we don't have to
    // allocate memory for a and z within the computeTheta function if we don't need them after.
    tmp_z_values.resize(total_nodes);
    tmp_a_values.resize(total_nodes);

    //Roll Drop outs
    //last dropped out for output node
    //add this one for not to break stuff
    while (hlayer_dropout_probs.size() < h_layer_num_nodes.size())
        hlayer_dropout_probs.push_back(0);
    rollDropouts();

    //fill frozen flags if not provided
    while (hlayer_is_frozen.size() < h_layer_num_nodes.size())
        hlayer_is_frozen.push_back(false);
}

void NNParam::initWeights(int init_type) {
    switch (init_type) {
        case PARAM_FULL_ZERO_INIT:
            fullZeroInit();
            break;
        case PARAM_ZERO_INIT:
            zeroInit();
            break;
        case PARAM_NORMAL_INIT:
            randomNormalInit();
            break;
        case NN_PARAM_VAR_SCALING_INIT:
            varianceScalingInit();
            break;
        case PARAM_RANDOM_INIT:
        default:
            randomUniformInit();
    }
}

//Randomly initialise all weights
void NNParam::randomUniformInit() {
    float min = -0.1, max = 0.1;
    // All Terms: to uniform values between -0.1 and 0.1 - Biases too
    std::uniform_real_distribution<float> distribution(min, max);
    for (unsigned int i = 0; i < weights.size(); i++)
        weights[i] = distribution(util_rng);//(float(std::rand()) / float(RAND_MAX) - 0.5) * 0.2;
}

void NNParam::randomNormalInit() {
    // All Terms: to normal values in mean and std
    float mean = 0.0f, std_dev = 0.05f, min = -0.1f, max = 0.1f;
    std::normal_distribution<float> distribution(mean, std_dev);

    unsigned int energy_length = getNumWeightsPerEnergyLevel();
    for (unsigned int energy_level_idx = 0; energy_level_idx < getNumEnergyLevels(); energy_level_idx++) {
        int weight_offset = 0;
        for (const auto &num_weights : num_weights_per_layer) {
            for (unsigned int i = 0; i < num_weights; i++) {
                float weight = 0;
                do {
                    weight = distribution(util_rng);
                } while ((weight < min) || (weight > max));
                weights[energy_length * energy_level_idx + weight_offset + i] = weight;
            }
            weight_offset += num_weights;
        }
    }
}

// Variance Scaling Initializer
void NNParam::varianceScalingInit() {
    float factor = 1.0f;
    float mean = -0.0f;

    // All Terms: to normal values in mean and std
    unsigned int energy_length = getNumWeightsPerEnergyLevel();
    for (unsigned int energy_level_idx = 0; energy_level_idx < getNumEnergyLevels(); energy_level_idx++) {
        int weight_offset = 0;
        for (int hlayer_idx = 0; hlayer_idx < h_layer_num_nodes.size(); ++hlayer_idx) {
            int fan_in = input_layer_node_num;
            if (hlayer_idx > 0)
                fan_in = h_layer_num_nodes[hlayer_idx - 1];
            int fan_out = h_layer_num_nodes[hlayer_idx];
            int num_weights = num_weights_per_layer[hlayer_idx];

            float std_dev = sqrt(factor / float(fan_out + fan_in));
            float min = -2 * std_dev, max = 2 * std_dev;
            std::normal_distribution<float> distribution(mean, std_dev);

            for (unsigned int i = 0; i < num_weights; i++) {
                float weight = 0;
                do {
                    weight = distribution(util_rng);
                } while ((weight < min) || (weight > max));
                weights[energy_length * energy_level_idx + weight_offset + i] = weight;
            }
            weight_offset += num_weights;
        }
    }
}

float NNParam::computeTheta(const FeatureVector &fv, int energy) {
    #ifdef _OPENMP
    // Thread-local scratch buffers to avoid data race on shared tmp_z_values/tmp_a_values
    thread_local azd_vals_t tl_z_values, tl_a_values;
    if (tl_z_values.size() != total_nodes) {
        tl_z_values.resize(total_nodes);
        tl_a_values.resize(total_nodes);
    }
    return computeTheta(fv, energy, tl_z_values, tl_a_values, true, false);
    #else
    return computeTheta(fv, energy, tmp_z_values, tmp_a_values, true, false);
    #endif
}

float NNParam::computeTheta(const FeatureVector &fv, int energy, azd_vals_t &z_values, azd_vals_t &a_values,
                             bool already_sized, bool use_dropout) {

    //Check Feature Length
    int fv_length = fv.getTotalLength();
    if( fv_length != expected_num_input_features ){
        std::cerr << "Expecting feature vector of length " << expected_num_input_features;
        std::cerr << " but found " << fv_length << std::endl;
        throw( ParamFeatureMismatchException() );
    }
    int energy_offset = getNumWeightsPerEnergyLevel() * energy;
    auto weights_it = weights.begin() + energy_offset;
    //Resize the z and a vectors to the required sizes
    if( !already_sized) {
        z_values.resize( total_nodes );
        a_values.resize( total_nodes );
    }

    //The first hidden layer takes the fv as input (which already has a bias feature,
    // so no additional biases in this layer)
    int neuron_idx = 0;
    auto layer_it = h_layer_num_nodes.begin();

    for (int h_node = 0; h_node < (*layer_it); h_node++) {
        if (!use_dropout || !hlayer_is_dropped[neuron_idx]) {
            float z_val = 0.0f;
            // for each feature, note feature_it is the index of feature
            // since we are using a sparse binary feature vector
            // each value is the index of value one
            for (auto feature_it = fv.getFeatureBegin(); feature_it != fv.getFeatureEnd(); ++feature_it)
                z_val += *(weights_it + *feature_it);
            weights_it += fv_length;
            // set input value z and active value a
            z_values[neuron_idx] = z_val;
            a_values[neuron_idx] = act_funcs[neuron_idx](z_val);
            // update act value if we are using drop out
            if (use_dropout)
                a_values[neuron_idx] = a_values[neuron_idx]/(1.0 - hlayer_dropout_probs[0]);

        } else if (hlayer_is_dropped[neuron_idx]){
            // if we are using drop out
            // and current node is dropped
            weights_it += fv_length;
            z_values[neuron_idx] = 0.0f;
            a_values[neuron_idx] = 0.0f;
        }
        neuron_idx++;
    }

    //Subsequent layers take the previous layer as input
    int num_input = h_layer_num_nodes[0];
    int input_layer_node_idx_start = 0;
    for (int h_layer_idx = 1; h_layer_idx < h_layer_num_nodes.size(); ++h_layer_idx) {
        for (int h_node_idx = 0; h_node_idx < h_layer_num_nodes[h_layer_idx]; ++h_node_idx) {
            if (!use_dropout || !hlayer_is_dropped[neuron_idx]) {
                float z_val = *weights_it++; //Bias

                // sum up and record z_val
                for (int i = 0; i < num_input; i++)
                    z_val += a_values[input_layer_node_idx_start + i] * (*weights_it++);

                // set input value z and active value a
                z_values[neuron_idx] = z_val;
                a_values[neuron_idx] = act_funcs[neuron_idx](z_val);

                // update output value if we are using drop out
                if (use_dropout)
                    a_values[neuron_idx] = a_values[neuron_idx]/(1.0 - hlayer_dropout_probs[h_layer_idx]);

            } else if (hlayer_is_dropped[neuron_idx]){
                // if dropped out move by num_input + 1
                // 1 for bais num_input weights
                weights_it += 1 + num_input;
                z_values[neuron_idx] = 0.0f;
                a_values[neuron_idx] = 0.0f;
            }
            neuron_idx ++;
        }

        input_layer_node_idx_start += num_input;
        num_input = h_layer_num_nodes[h_layer_idx];
    }

    return a_values.back();    //The output of the last layer is theta
}

void NNParam::saveToFile(std::string &filename) {

    //Write all the weights etc using the parent function
    Param::saveToFile(filename);

    //Re-open the file and add the neural net configuration parameters
    std::ofstream out;
    out.open(filename.c_str(), std::fstream::app);
    if (!out.is_open()) {
        std::cout << "Warning: Trouble opening parameter file" << std::endl;
    } else {
        out << "Neural Net Config" << std::endl;
        out << h_layer_num_nodes.size() << std::endl;
        std::vector<int>::iterator iit = h_layer_num_nodes.begin();
        for (; iit != h_layer_num_nodes.end(); ++iit)
            out << *iit << " ";
        out << std::endl;
        out << act_func_ids.size() << std::endl;
        iit = act_func_ids.begin();
        // output activation function for each layer
        for (; iit != act_func_ids.end(); ++iit)
            out << *iit << " ";
        out << std::endl;
        // output dropout probs for each layer
        for (const auto &prob : hlayer_dropout_probs)
            out << prob << " ";
        out << std::endl;
        // output frozen flag for each layer
        for (const auto &is_frozen : hlayer_is_frozen)
            out << is_frozen << " ";
        out << std::endl;

        out.close();
    }
}

NNParam::NNParam(std::string &filename) : Param(filename) {

    //We've already read all the weights, here we want to read the neural net parameters,
    //which should be appended right at the end of the file
    std::string line;
    std::ifstream ifs(filename.c_str(), std::ifstream::in);

    bool found_nn_details = false;
    if (!ifs) std::cout << "Could not open file " << filename << std::endl;
    while (getline(ifs, line)) {
        if (line.size() > 6 && line.substr(0, 6) == "Neural") {
            //Get the hidden node number configuration
            getline(ifs, line);
            total_nodes = 0;
            int num_layers = std::stoi(line);
            h_layer_num_nodes.resize(num_layers);
            getline(ifs, line);
            std::stringstream ss1(line);
            for (int i = 0; i < num_layers; i++) {
                ss1 >> h_layer_num_nodes[i];
                total_nodes += h_layer_num_nodes[i];
            }
            //Get the activation function configuration
            getline(ifs, line);
            int act_func_size = std::stoi(line);
            act_func_ids.resize(act_func_size);
            getline(ifs, line);
            std::stringstream ss2(line);
            for (int i = 0; i < num_layers; i++)
                ss2 >> act_func_ids[i];
            setActivationFunctionsFromIds();

            // Get the dropout probs for each layer
            hlayer_dropout_probs.resize(act_func_size);
            getline(ifs, line);
            if(!ifs.eof()){
                std::stringstream ss3(line);
                // get  and roll drop outs
                for (int i = 0; i < num_layers; i++)
                    ss3 >> hlayer_dropout_probs[i];
            }
            rollDropouts();

            // Get the frozen flag for each layer
            hlayer_is_frozen.resize(act_func_size);
            getline(ifs, line);
            if(!ifs.eof()){
                std::stringstream ss4(line);
                // get  and roll drop outs
                for (int i = 0; i < num_layers; ++i)
                    ss4 >> hlayer_is_frozen[i];
            }

            tmp_z_values.resize(total_nodes);
            tmp_a_values.resize(total_nodes);
            found_nn_details = true;
            break;
        }
    }
    ifs.close();

    if (!found_nn_details) throw NNParamFileReadException();
}

void NNParam::setActivationFunctionsFromIds() {

    std::vector<int>::iterator it, itt;
    itt = h_layer_num_nodes.begin();
    for (it = act_func_ids.begin(); it != act_func_ids.end(); ++it, ++itt) {
        for (int hnode = 0; hnode < *itt; hnode++) {
            if (*it == LINEAR_NN_ACTIVATION_FUNCTION) {
                act_funcs.push_back(linear_activation);
                deriv_funcs.push_back(linear_derivative);
            } else if (*it == RELU_AND_NEG_RLEU_NN_ACTIVATION_FUNCTION) {
                if (hnode % 2 == 1) {
                    act_funcs.push_back(neg_relu_activation);
                    deriv_funcs.push_back(neg_relu_derivative);
                } else {
                    act_funcs.push_back(relu_activation);
                    deriv_funcs.push_back(relu_derivative);
                }
            } else if (*it == LEAKY_RELU_NN_ACTIVATION_FUNCTION) {
                act_funcs.push_back(leaky_relu_activation);
                deriv_funcs.push_back(leaky_relu_derivative);
            } else if (*it == RELU_NN_ACTIVATION_FUNCTION) {
                act_funcs.push_back(relu_activation);
                deriv_funcs.push_back(relu_derivative);
            } else throw NNParamActivationFunctionIdException();
        }
    }

}

void NNParam::computeDeltas(std::vector<azd_vals_t> &deltasA, std::vector<azd_vals_t> &deltasB,
                            std::vector<azd_vals_t> &z_values, std::vector<azd_vals_t> &a_values, float rho_denom,
                            int energy) {

    //Resize the output delta vectors
    unsigned int num_trans_from_id = z_values.size();
    deltasA.resize(num_trans_from_id);
    deltasB.resize(num_trans_from_id);
    for (int idx = 0; idx < num_trans_from_id; idx++) {
        deltasA[idx].resize(total_nodes);
        deltasB[idx].resize(total_nodes);
    }

    //Set up the iterators (we are going backwards through the network, so set up iterators in reverse)
    std::vector<azd_vals_t::reverse_iterator> itAs(num_trans_from_id), itBs(num_trans_from_id);
    std::vector<azd_vals_t::reverse_iterator> aits(num_trans_from_id), zits(num_trans_from_id);
    for (int idx = 0; idx < num_trans_from_id; idx++) {
        aits[idx] = a_values[idx].rbegin();
        zits[idx] = z_values[idx].rbegin();
        itAs[idx] = deltasA[idx].rbegin();
        itBs[idx] = deltasB[idx].rbegin();
    }
    std::vector<float (*)(float)>::reverse_iterator itdf = deriv_funcs.rbegin();
    std::vector<int>::reverse_iterator itlayer = h_layer_num_nodes.rbegin();
    std::vector<azd_vals_t::reverse_iterator> itAs_input = itAs, itBs_input = itBs;

    int energy_offset = (num_energy_levels - energy - 1) * getNumWeightsPerEnergyLevel();
    std::vector<float>::reverse_iterator wit = weights.rbegin() + energy_offset;

    //Compute the highest level deltas
    for (int idx = num_trans_from_id - 1; idx >= 0; idx--) {
        float deriv_val = (*itdf)(*(zits[idx])++);
        float theta = (*(aits[idx])++);
        float rho_val = exp(theta) / rho_denom;
        *(itAs[idx])++ = deriv_val;
        *(itBs[idx])++ = deriv_val * rho_val;
    }
    itlayer++;
    itdf++;
    int num_output = 1;

    //Compute the deltas for the subsequent layers
    std::vector<float>::reverse_iterator itA_input_tmp, itB_input_tmp, wit_tmp;
    for (; itlayer != h_layer_num_nodes.rend(); ++itlayer) {
        std::vector<float (*)(float)>::reverse_iterator itdf_tmp;
        for (int idx = num_trans_from_id - 1; idx >= 0; idx--) {
            itdf_tmp = itdf;
            for (int hnode_r = 0; hnode_r < *itlayer; hnode_r++) {
                wit_tmp = wit + hnode_r;
                itA_input_tmp = itAs_input[idx];
                itB_input_tmp = itBs_input[idx];
                float Aval = 0.0, Bval = 0.0;
                float deriv_val = (*itdf_tmp)(*zits[idx]++);
                for (int oidx = num_output - 1; oidx >= 0; oidx--) {
                    float w_deriv_val = (*wit_tmp) * deriv_val;
                    Aval += (*itA_input_tmp++) * w_deriv_val;
                    Bval += (*itB_input_tmp++) * w_deriv_val;
                    wit_tmp += (oidx > 0) * (*itlayer + 1);
                }
                *(itAs[idx])++ = Aval;
                *(itBs[idx])++ = Bval;
                itdf_tmp++;
            }
            itAs_input[idx] = itA_input_tmp;
            itBs_input[idx] = itB_input_tmp;
            wit_tmp++;
            wit_tmp++;    //Skip the bias
        }
        num_output = *itlayer;
        itdf = itdf_tmp;
        wit = wit_tmp;
    }
}

void NNParam::computeUnweightedGradients(std::vector<std::vector<float> > &unweighted_grads,
                                         std::set<unsigned int> &used_idxs, std::vector<const FeatureVector *> &fvs,
                                         std::vector<azd_vals_t> &deltasA, std::vector<azd_vals_t> &deltasB,
                                         std::vector<azd_vals_t> &a_values) {

    unsigned int num_trans_from_id = a_values.size();
    unweighted_grads.resize(num_trans_from_id + 1); //+1 for the persistence transition (stored last)
    std::vector<std::vector<float>::iterator> itgrads(num_trans_from_id), itAs(num_trans_from_id), itBs(
            num_trans_from_id), aits(num_trans_from_id);
    for (int idx = 0; idx < num_trans_from_id; idx++) {
        unweighted_grads[idx].clear();
        unweighted_grads[idx].resize(getNumWeightsPerEnergyLevel(), 0.0f);
        itgrads[idx] = unweighted_grads[idx].begin();
        aits[idx] = a_values[idx].begin();
        itAs[idx] = deltasA[idx].begin();
        itBs[idx] = deltasB[idx].begin();
    }

    //Persistence (and normalization) terms
    unweighted_grads[num_trans_from_id].clear();
    unweighted_grads[num_trans_from_id].resize(getNumWeightsPerEnergyLevel(), 0.0f);

    //If there's no transitions from this id except persistence, the unweighted gradients for that persistence transition will all be 0
    if (num_trans_from_id == 0) return;

    std::vector<float>::iterator normit = unweighted_grads[num_trans_from_id].begin();
    unsigned int feature_len = fvs[0]->getTotalLength();

    //Collect the used feature idxs (we'll need these for the first layer normalization)
    int frozen_layer_idx = 0;
    bool froze_layer = hlayer_is_frozen[frozen_layer_idx];
    std::set<unsigned int> tmp_used_idxs;
    std::vector<unsigned int> tmp_used_idxs_vect(feature_len);
    if (!froze_layer){
        for (int idx = 0; idx < num_trans_from_id; idx++) {
            for (auto fit = fvs[idx]->getFeatureBegin(); fit != fvs[idx]->getFeatureEnd(); ++fit) {
                //tmp_used_idxs.insert(*fit);
                tmp_used_idxs_vect[*fit] ++;
                for (int hnode = 0; hnode < h_layer_num_nodes[0]; hnode++)
                    used_idxs.insert(hnode * feature_len + *fit);
            }
        }
    }

    //First layer (only compute gradients for indices relevant to used features)
    std::vector<int>::iterator itlayer = h_layer_num_nodes.begin();

    for (int hnode = 0; hnode < *itlayer; hnode++) {
        //Compute the unnormalized terms, tracking the normalizers (persistence terms) as we go
        if (!froze_layer){
            for (int idx = 0; idx < num_trans_from_id; idx++) {
                float deltaA = *itAs[idx]++;
                float deltaB = *itBs[idx]++;
                for (auto fit = fvs[idx]->getFeatureBegin(); fit != fvs[idx]->getFeatureEnd(); ++fit) {
                    *(itgrads[idx] + *fit) = deltaA;
                    *(normit + *fit) -= deltaB;
                }
            }
            // Apply the normalizers
            // (Note: we need to normalize for all transitions if any transition used that feature)
            for (int idx = 0; idx < num_trans_from_id; idx++) {
                /*std::set<unsigned int>::iterator sit = tmp_used_idxs.begin();
                for (; sit != tmp_used_idxs.end(); ++sit)
                    *(itgrads[idx] + *sit) += *(normit + *sit);*/
                for(int fv_idx = 0; fv_idx < tmp_used_idxs_vect.size(); ++fv_idx)
                    if(tmp_used_idxs_vect[fv_idx] > 0 )
                        *(itgrads[idx] + fv_idx) += *(normit + fv_idx);
                itgrads[idx] += feature_len;
            }
        }
        // keep update normit regard if we update or not
        normit += feature_len;
    }

    //Subsequent layers
    int num_input = *itlayer++;
    for (; itlayer != h_layer_num_nodes.end(); ++itlayer) {
        frozen_layer_idx += 1;
        froze_layer = hlayer_is_frozen[frozen_layer_idx];
        
        for (int hnode = 0; hnode < *itlayer; hnode++) {
            if (!froze_layer) {
                //Compute the unnormalized terms, tracking the normalizers (persistence terms) as we go
                for (int idx = 0; idx < num_trans_from_id; idx++) {
                    float deltaA = *itAs[idx]++;
                    float deltaB = *itBs[idx]++;
                    *itgrads[idx] = deltaA; //Bias term
                    *normit -= deltaB;
                    for (int input_node = 0; input_node < num_input; ++input_node) {
                        float a_val = *(aits[idx] + input_node);
                        *(itgrads[idx] + input_node + 1) = deltaA * a_val;
                        *(normit + input_node + 1) -= deltaB * a_val;
                    }
                }

                //Apply the normalizers
                for (int idx = 0; idx < num_trans_from_id; idx++) {
                    *itgrads[idx] += *normit; //Bias term
                    for (int input_node = 0; input_node < num_input; ++input_node)
                        *(itgrads[idx] + input_node + 1) += *(normit + input_node + 1);
                    itgrads[idx] += (num_input + 1);
                }
            }

            normit += (num_input + 1);
        }

        for (int idx = 0; idx < num_trans_from_id; idx++) 
            aits[idx] += num_input;
        num_input = *itlayer;
    }

}

void NNParam::getBiasIndexes(std::vector<unsigned int> &bias_indexes) {

    bias_indexes.resize(total_nodes);
    unsigned int idx = 0, count = 0;
    auto itlayer = h_layer_num_nodes.begin();
    // Adding bias index for input layer
    for (int hnode = 0; hnode < h_layer_num_nodes[0]; hnode++, count++) {
        bias_indexes[count] = idx;
        idx += expected_num_input_features;
    }

    itlayer ++;
    // Adding bias index for hidden layers
    for (; itlayer != h_layer_num_nodes.end(); ++itlayer) {
        for (int hnode = 0; hnode < *itlayer; hnode++, count++) {
            bias_indexes[count] = idx;
            idx += (*(itlayer - 1) + 1);
        }
    }
}

void NNParam::rollDropouts() {
    if (hlayer_is_dropped.empty())
        hlayer_is_dropped.resize(total_nodes);
    // set everything to false
    std::fill(hlayer_is_dropped.begin(),hlayer_is_dropped.end(), false);

    //adding dropouts in the all layer with the same prob
    int neuron_idx = 0;
    for (int hlayer_idx = 0; hlayer_idx < h_layer_num_nodes.size(); ++hlayer_idx) {
        // P(b|p) = p if b == true
        // P(b|p) = 1-p if b == false
        float drop_ratio = hlayer_dropout_probs[hlayer_idx];// + iter * delta;

        std::vector<int> indice;
        for (int h_node_idx = 0; h_node_idx < h_layer_num_nodes[hlayer_idx]; ++h_node_idx){
            indice.push_back(neuron_idx);
            neuron_idx ++;
        }

        auto expected_deactive_count = (unsigned int)std::floor(drop_ratio * h_layer_num_nodes[hlayer_idx]);
        std::shuffle(indice.begin(),indice.end(),util_rng);
        indice.resize(expected_deactive_count);

        for (const auto & idx: indice)
            hlayer_is_dropped[idx] = true;
    }
}