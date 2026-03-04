/*#########################################################################
# Mass Spec Prediction and Identification of Metabolites
#
# main.cpp
#
# Description:   Predict the MS/MS spectra for a given structure using a
#                pre-trained CFM model.
#
# Copyright (c) 2013, Felicity Allen
# All rights reserved.

# This file is part of the cfm-id project.
# The contents are covered by the terms of the GNU Lesser General Public
# License, which is included in the file license.txt, found at the root
# of the cfm source tree.
#
# Modified: OpenMP parallelization of batch prediction
#########################################################################*/

#include "Config.h"
#include "Param.h"
#include "MolData.h"
#include "Version.h"

#include <GraphMol/SanitException.h>
#include <GraphMol/SmilesParse/SmilesParse.h>
#include <boost/filesystem.hpp>

#ifdef _OPENMP
#include <omp.h>
#endif

#include <stdexcept>

int main(int argc, char *argv[]);

class SpectrumPredictionException : public std::exception {
private:
    std::runtime_error message_;
public:
    SpectrumPredictionException(const std::string & message) noexcept: message_(message) {};

    const char *what() const noexcept override {
        return message_.what();
    }

    ~SpectrumPredictionException() noexcept override = default;;
};

class FileException : public std::exception {
private:
    std::runtime_error message_;
public:
    FileException(const std::string & message) noexcept: message_(message) {};

    const char *what() const noexcept override {
        return message_.what();
    }

    ~FileException() noexcept override = default;;
};


void parseInputFile(std::vector<MolData> &data, std::string &input_filename, config_t *cfg);

int main(int argc, char *argv[]) {
    // Disable stdout buffering for real-time Docker log output
    std::cout << std::unitbuf;

    bool to_stdout = true;
    int do_annotate = 0;
    int postprocessing_method = 1;
    int suppress_exceptions = 1;
    std::string output_filename;
    std::string param_filename = "param_output.log";
    std::string config_filename = "param_config.txt";
    double prob_thresh_for_prune = 0.001;
    double postprocessing_energy = -1;
    int min_peaks = -1;
    int max_peaks = -1;
    double min_peak_intensity = -1;
    std::string single_prediction_id = "NullId";

    if (argc != 6 && argc != 2 && argc != 5 && argc != 3 && argc != 7 && argc != 8 && argc != 9 && argc != 10 &&
        argc != 11 && argc != 12 && argc != 13 && argc != 14) {
        std::cout << std::endl << std::endl;
        std::cout << std::endl
                  << "CFM-ID Version: " << PROJECT_VER << std::endl
                  << "Usage: cfm-predict <input_smiles_or_inchi> <prob_thresh_for_prune> <param_filename> <config_filename> <include_annotations> <output_filename> <apply_post_processing>"
                  << std::endl << std::endl << std::endl;
        std::cout << std::endl << "input_smiles_or_inchi_or_file:" << std::endl
                  << "The smiles or inchi string of the structure whose spectra you want to predict, or a .txt file containing a list of <id smiles> pairs, one per line."
                  << std::endl;
        std::cout << std::endl << "prob_thresh_for_prune (opt):" << std::endl
                  << "The probability below which to prune unlikely fragmentations (default 0.001)" << std::endl;
        std::cout << std::endl << "param_filename (opt):" << std::endl
                  << "The filename where the parameters of a trained cfm model can be found (if not given, assumes param_output.log in current directory)"
                  << std::endl;
        std::cout << std::endl << "config_filename (opt):" << std::endl
                  << "The filename where the configuration parameters of the cfm model can be found (if not given, assumes param_config.txt in current directory)"
                  << std::endl;
        std::cout << std::endl << "include_annotations (opt):" << std::endl
                  << "Whether to include fragment information in the output spectra (0 = NO (default), 1 = YES ). Note: ignored for msp/mgf output."
                  << std::endl;
        std::cout << std::endl << "output_filename_or_dir (opt):" << std::endl
                  << "The filename of the output spectra file to write to (if not given or given stdout, prints to stdout), OR directory if multiple smiles inputs are given (else current directory) OR msp or mgf file."
                  << std::endl;
        std::cout << std::endl << "postprocessing method (opt):" << std::endl
                  << "Post-process predicted spectra with: "
                     "1.Use settings specified in config file, if no fall back to OPT#2 \n"
                     "2.take the top 80% of energy, or the highest 30 peaks (whichever comes first) \n "
                     "(0 = OFF, 1 = OPT#1 (default) if not specified fallback to #2, 2 = OPT#2)."
                  << std::endl;
        std::cout << std::endl << "suppress_exception (opt):" << std::endl
                  << "Suppress exceptions so that the program returns normally even when it fails to produce a result (0 = OFF , 1 = ON (default))."
                  << std::endl;
        std::cout << std::endl << "postprocessing_energy (opt):" << std::endl
                  << "postprocessing energy out of 80% (default 80%)"
                  << std::endl;
        std::cout << std::endl << "min_peak_intensity [0,100.0] (opt):" << std::endl
                  << "min amount of peak relative intensity, enter -1 to use default" << std::endl;
        std::cout << std::endl << "override min peaks constraint (opt):" << std::endl
                  << "min amount of peak will include in the spectra, enter -1 to use default"
                  << std::endl;
        std::cout << std::endl << "override max peaks constraint(opt):" << std::endl
                  << "max amount of peak will include in the spectra, enter -1 to use default"
                  << std::endl;
        std::cout << std::endl << "prediction id (opt):" << std::endl
                  << "id for predicted spectra, only used in single input mode"
                  << std::endl;
        exit(1);
    }

    std::string input_smiles_or_inchi = argv[1];
    if (argc >= 3) {
        try { prob_thresh_for_prune = boost::lexical_cast<float>(argv[2]); }
        catch (boost::bad_lexical_cast &e) {
            std::cout << "Invalid prob_thresh_for_prune: " << argv[2] << std::endl;
            exit(1);
        }
    }
    if (argc >= 5) {
        param_filename = argv[3];
        config_filename = argv[4];
    }
    if (argc >= 6) {
        try { do_annotate = boost::lexical_cast<bool>(argv[5]); }
        catch (boost::bad_lexical_cast &e) {
            std::cout << "Invalid include_annotations (Must be 0 or 1): " << argv[5] << std::endl;
            exit(1);
        }
    }
    if (argc >= 7) {
        output_filename = argv[6];
        if(output_filename != "stdout")
            to_stdout = false;
    }
    if (argc >= 8) {
        try { postprocessing_method = boost::lexical_cast<int>(argv[7]); }
        catch (boost::bad_lexical_cast &e) {
            std::cout << "Invalid postprocessing_method (Must be 0, 1 or 2): " << argv[7] << std::endl;
            exit(1);
        }
        if (postprocessing_method == 0) {
            min_peaks = 1;
            max_peaks = 1000;
            postprocessing_energy = 100;
        }
        if (postprocessing_method == 2) {
            min_peaks = 1;
            max_peaks = 30;
            postprocessing_energy = 80;
        }
    }
    if (argc == 9) {
        try { suppress_exceptions = boost::lexical_cast<bool>(argv[8]); }
        catch (boost::bad_lexical_cast &e) {
            std::cout << "Invalid suppress_exceptions (Must be 0 or 1): " << argv[8] << std::endl;
            exit(1);
        }
    }
    if (argc == 10) {
        try { postprocessing_energy = boost::lexical_cast<double>(argv[9]); }
        catch (boost::bad_lexical_cast &e) {
            std::cout << "Invalid postprocessing_energy (Must be 0 or 100): " << argv[9] << std::endl;
            exit(1);
        }
    }

    if (argc == 11) {
        try {
            auto input_min_peak_intensity = boost::lexical_cast<double>(argv[10]);
            if (input_min_peak_intensity >= 0)
                min_peak_intensity = input_min_peak_intensity;
        }
        catch (boost::bad_lexical_cast &e) {
            std::cout << "Invalid min_peaks: " << argv[10] << std::endl;
            exit(1);
        }
    }

    if (argc == 12) {
        try {
            auto input_min_peaks = boost::lexical_cast<int>(argv[11]);
            if (input_min_peaks >= 0)
                min_peaks = input_min_peaks;
        }
        catch (boost::bad_lexical_cast &e) {
            std::cout << "Invalid min peaks: " << argv[11] << std::endl;
            exit(1);
        }
    }

    if (argc == 13) {
        try {
            auto input_max_peaks = boost::lexical_cast<int>(argv[12]);
            if (input_max_peaks >= 0)
                max_peaks = input_max_peaks;
        }
        catch (boost::bad_lexical_cast &e) {
            std::cout << "Invalid max peaks: " << argv[12] << std::endl;
            exit(1);
        }
    }

    if (argc == 14) {
        single_prediction_id = argv[13];
    }

    //Initialise model configuration
    config_t cfg;
    if (!boost::filesystem::exists(config_filename)) {
        std::cout << "Could not find file: " << config_filename << std::endl;
        if (!suppress_exceptions)
            throw FileException("Could not find file: " + param_filename);
        exit(1);
    }
    initConfig(cfg, config_filename);

    if (postprocessing_method == 1){
        postprocessing_energy = postprocessing_energy < 0 ? cfg.default_postprocessing_energy : postprocessing_energy;
        min_peaks = min_peaks == -1 ? cfg.default_predicted_peak_min : min_peaks;
        max_peaks = max_peaks == -1 ? cfg.default_predicted_peak_max : max_peaks;
        min_peak_intensity = min_peak_intensity < 0 ? cfg.default_predicted_min_intensity : min_peak_intensity;
    }

    //Read in the parameters
    if (!boost::filesystem::exists(param_filename)) {
        std::cout << "Could not find file: " << param_filename << std::endl;
        if (!suppress_exceptions)
            throw FileException("Could not find file: " + param_filename);
        exit(1);
    }

    Param *param;
    NNParam *nn_param;
    if (cfg.theta_function == NEURAL_NET_THETA_FUNCTION)
        nn_param = new NNParam(param_filename);
    else
        param = new Param(param_filename);

    //Check for mgf or msp output - and setup in exists
    int output_mode = NO_OUTPUT_MODE;
    std::ostream *out;
    std::ofstream of;
    std::streambuf *buf;
    if (!to_stdout && output_filename.substr(output_filename.size() - 4) == ".msp") {
        output_mode = MSP_OUTPUT_MODE;
        of.open(output_filename.c_str());
        if (!of.is_open()) {
            std::cerr << "Error: Could not open output msp file " << output_filename << std::endl;
            if (!suppress_exceptions)
                throw FileException("Could not open output msp file " + output_filename);
        }
        buf = of.rdbuf();
        out = new std::ostream(buf);
    } else if (!to_stdout && output_filename.substr(output_filename.size() - 4) == ".mgf") {
        output_mode = MGF_OUTPUT_MODE;
        of.open(output_filename.c_str());
        if (!of.is_open()) {
            std::cerr << "Error: Could not open output mgf file " << output_filename << std::endl;
            if (!suppress_exceptions)
                throw FileException("Could not open output mgf file " + output_filename);
        }
        buf = of.rdbuf();
        out = new std::ostream(buf);
    } else if (!to_stdout &&
        (output_filename.substr(output_filename.size() - 4) == ".txt"
        ||  output_filename.substr(output_filename.size() - 4) == ".log")) {
        output_mode = SINGLE_TXT_OUTPUT_MODE;
        of.open(output_filename.c_str());
        if (!of.is_open()) {
            std::cerr << "Error: Could not open output txt/log file " << output_filename << std::endl;
            if (!suppress_exceptions)
                throw FileException("Could not open output txt/log file " + output_filename);
        }
        buf = of.rdbuf();
        out = new std::ostream(buf);
    }
    //Check for batch input - if found, read in inchis and set up output directory, mgf or msp
    std::vector<MolData> data;
    bool batch_run = false;
    std::string output_dir_str;
    if (input_smiles_or_inchi.size() > 4 &&
        input_smiles_or_inchi.substr(input_smiles_or_inchi.size() - 4) == ".txt") {

        try {
            parseInputFile(data, input_smiles_or_inchi, &cfg);
        }
        catch (FileException &e) {
            if (!suppress_exceptions)
                throw FileException("Could not parse input file " + input_smiles_or_inchi);
            exit(1);
        }
        batch_run = true;
        if (!to_stdout && output_mode == NO_OUTPUT_MODE) {
            if (output_filename != "." && !boost::filesystem::exists(output_filename))
                boost::filesystem::create_directory(output_filename);
            output_dir_str = output_filename + "/";
        }
        if (!to_stdout)
            std::cout << "Read " << data.size() << " molecules from input file." << std::endl;
    } else
        data.push_back(MolData(single_prediction_id.c_str(), input_smiles_or_inchi.c_str(), &cfg));

    // ═══════════════════════════════════════════════════════════════
    // OpenMP PARALLELIZED batch prediction
    // Each molecule's prediction is independent:
    //   - fragment graph computation (CPU-intensive)
    //   - spectrum prediction
    //   - file output (each molecule writes to its own .log file)
    // Only parallelize batch mode with per-file output (NO_OUTPUT_MODE)
    // Single file outputs (MSP/MGF/TXT) remain sequential
    // ═══════════════════════════════════════════════════════════════
    bool can_parallelize = batch_run && output_mode == NO_OUTPUT_MODE && !to_stdout;
    int total_mols = (int)data.size();

#ifdef _OPENMP
    if (can_parallelize) {
        std::cout << "[Predict] OpenMP batch prediction: " << total_mols
                  << " molecules, " << omp_get_max_threads() << " threads" << std::endl;
    }
#endif

    int progress_done = 0;

#ifdef _OPENMP
    #pragma omp parallel for schedule(dynamic) if(can_parallelize)
#endif
    for (int mol_idx = 0; mol_idx < total_mols; ++mol_idx) {
        // Each thread works on its own copy of MolData
        MolData mol_data = data[mol_idx];

        try {
            // Calculate the pruned FragmentGraph
            LikelyFragmentGraphGenerator *fgen;
            if (cfg.theta_function == NEURAL_NET_THETA_FUNCTION)
                fgen = new LikelyFragmentGraphGenerator(nn_param, &cfg, prob_thresh_for_prune);
            else
                fgen = new LikelyFragmentGraphGenerator(param, &cfg, prob_thresh_for_prune);

            mol_data.computeLikelyFragmentGraphAndSetThetas(*fgen, prob_thresh_for_prune, do_annotate);
            mol_data.computePredictedSpectra(*nn_param, true, -1, min_peaks, max_peaks, postprocessing_energy, min_peak_intensity, cfg.use_log_scale_peak);
            delete fgen;
        }
        catch (RDKit::MolSanitizeException &e) {
            #pragma omp critical(stderr_output)
            std::cerr << "Could not sanitize input: "  << mol_data.getId() << " " << mol_data.getSmilesOrInchi() << std::endl;
            if (!batch_run && !suppress_exceptions)
                throw SpectrumPredictionException("RDKit could not sanitize input: " + mol_data.getSmilesOrInchi());
            continue;
        }
        catch (RDKit::SmilesParseException &pe) {
            #pragma omp critical(stderr_output)
            std::cerr << "Could not parse input: "  << mol_data.getId() << " " << mol_data.getSmilesOrInchi() << std::endl;
            if (!batch_run && !suppress_exceptions)
                throw SpectrumPredictionException("RDKit could not parse input: " + mol_data.getSmilesOrInchi());
            continue;
        }
        catch (FragmentGraphGenerationException &ge) {
            #pragma omp critical(stderr_output)
            std::cerr << "Could not compute fragmentation graph for input: "  << mol_data.getId() << " " << mol_data.getSmilesOrInchi() << std::endl;
            if (!batch_run && !suppress_exceptions)
                throw SpectrumPredictionException(
                        "Could not compute fragmentation graph for input: " + mol_data.getSmilesOrInchi());
            continue;
        }
        catch (IonizationException &ie) {
            #pragma omp critical(stderr_output)
            std::cerr << "Could not ionize: " << mol_data.getId() << " " << mol_data.getSmilesOrInchi() << std::endl;
            if (!batch_run && !suppress_exceptions) throw IonizationException();
            continue;
        }
        catch (FragmentGraphTimeoutException &te) {
            #pragma omp critical(stderr_output)
            std::cerr << "Timeout computing fragmentation graph for input: "  << mol_data.getId() << " " << mol_data.getSmilesOrInchi() << std::endl;
            continue;
        }
        catch (std::runtime_error &e) {
            #pragma omp critical(stderr_output)
            std::cerr << e.what() << std::endl;
            if (!batch_run && !suppress_exceptions)
                throw std::runtime_error(e.what());
            continue;
        }

        // Write output — thread-safe for per-file output mode
        if (can_parallelize) {
            // Each molecule writes to its own file — no locking needed
            std::string mol_output_filename = output_dir_str + mol_data.getId() + ".log";
            std::ofstream mol_of(mol_output_filename.c_str());
            if (mol_of.is_open()) {
                std::streambuf *mol_buf = mol_of.rdbuf();
                std::ostream mol_out(mol_buf);
                mol_data.outputSpectra(mol_out, "Predicted", do_annotate);
                mol_of.close();
            }

            // Thread-safe progress reporting
            #pragma omp critical(progress_output)
            {
                ++progress_done;
                std::cout << "[Predict] " << progress_done << "/" << total_mols
                          << " " << mol_data.getId() << std::endl;
            }
        } else {
            // Original sequential output logic
            if (output_mode == NO_OUTPUT_MODE) {
                if (!to_stdout) {
                    if (batch_run) output_filename = output_dir_str + mol_data.getId() + ".log";
                    of.open(output_filename.c_str());
                    if (!of.is_open()) {
                        std::cerr << "Error: Could not open output file " << output_filename << std::endl;
                        if (!batch_run && !suppress_exceptions)
                            throw FileException("Could not open output file " + output_filename);
                    }
                    buf = of.rdbuf();
                } else buf = std::cout.rdbuf();
                out = new std::ostream(buf);
            }

            if (output_mode == NO_OUTPUT_MODE || output_mode == SINGLE_TXT_OUTPUT_MODE) {
                if (output_mode == SINGLE_TXT_OUTPUT_MODE && mol_idx > 0 )
                    *out << std::endl << std::endl;
                mol_data.outputSpectra(*out, "Predicted", do_annotate);
            } else if (output_mode == MSP_OUTPUT_MODE)
                mol_data.writePredictedSpectraToMspFileStream(*out);
            else if (output_mode == MGF_OUTPUT_MODE)
                mol_data.writePredictedSpectraToMgfFileStream(*out);

            if (output_mode == NO_OUTPUT_MODE) {
                if (!to_stdout) of.close();
                delete out;
            }

            if (!to_stdout)
                std::cout << "("<< mol_idx + 1 <<"/" << data.size()<< ") Predicted Spectra for " << mol_data.getId() << " " << mol_data.getSmilesOrInchi() << std::endl;
        }
    }
    if (output_mode != NO_OUTPUT_MODE) delete out;
    return (0);
}

void parseInputFile(std::vector<MolData> &data, std::string &input_filename, config_t *cfg) {

    std::string line, smiles_or_inchi, id;
    std::ifstream ifs(input_filename.c_str(), std::ifstream::in);

    //Get the first line - the number of input molecules
    if (!ifs.good()) {
        std::cout << "Could not open input file: " << input_filename << std::endl;
        throw FileException("Could not open input file: " + input_filename);
    }

    //Now get all the molecules
    int i = 0;
    while (ifs.good()) {
        i++;

        getline(ifs, line);
        if (line.size() < 3) continue;

        std::stringstream ss(line);
        ss >> id >> smiles_or_inchi;

        data.push_back(MolData(id.c_str(), smiles_or_inchi.c_str(), cfg));
    }
}
