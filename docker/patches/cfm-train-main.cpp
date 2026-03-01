/*#########################################################################
# Mass Spec Prediction and Identification of Metabolites
#
# main.cpp
#
# Description: 	Learn the parameters of a model for the mass spec
#				fragmentation process using EM, then use it to predict
#				the mass spectra.
#
# Copyright (c) 2013, Felicity Allen
# All rights reserved.

# This file is part of the cfm-id project.
# The contents are covered by the terms of the GNU Lesser General Public
# License, which is included in the file license.txt, found at the root
# of the cfm source tree.
#
# Modified: OpenMP parallelization of fragment graph generation
#########################################################################*/

#include "mpi.h"

#include <boost/algorithm/string.hpp>
#include <boost/lexical_cast.hpp>
#include <boost/filesystem.hpp>
#include <boost/program_options.hpp>

#ifdef _OPENMP
#include <omp.h>
#endif

#include "EmModel.h"
#include "EmNNModel.h"

void parseInputFile(std::vector<MolData> &data, std::string &input_filename, int mpi_rank, int mpi_nump, config_t *cfg);

void
trainSingleEnergyCFM(std::string &param_filename, config_t &cfg, FeatureCalculator &fc, std::string &status_filename,
                     int group, std::vector<MolData> &data, int start_energy, int no_train, int start_repeat);

int main(int argc, char *argv[]) {
    // Disable stdout buffering so Docker logs show progress in real-time
    std::cout << std::unitbuf;

    int mpi_rank, mpi_nump;
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &mpi_rank);
    MPI_Comm_size(MPI_COMM_WORLD, &mpi_nump);

    // Define parameters
    std::string input_filename;   //List (one per line): id, smiles_or_inchi, group
    std::string feature_filename; //List of features, line-spaced
    std::string config_filename;    //Parameter configuration
    std::string peakfile_dir_or_msp;    //MSP file or Directory containing the peak files for each molecule (in format <id>.txt)
    std::string data_folder;
    std::string status_filename;
    std::string fv_fragment_graphs_folder;

    bool no_train = false;
    int start_energy = 0, start_repeat = 0;

    //Cross validation groups to process
    int min_group = 0, max_group = 0;

    //Define and parse the program options
    namespace po = boost::program_options;
    po::options_description general_options("Uasge");
    general_options.add_options()
            ("help,h", "Help message")
            ("input_filename,i", po::value<std::string>(&input_filename)->required(),
             "Text file with number of mols on first line, "
             "then id smiles_or_inchi cross_validation_group on each line after that.")
            ("feature_filename,f", po::value<std::string>(&feature_filename)->required(),
             "Text file with list of feature names to include, line separated")
            ("config_filename,c", po::value<std::string>(&config_filename)->required(),
             "Text file listing configuration parameters."
             " Line separated 'name value'.")
            ("peakfile_dir_or_msp,p", po::value<std::string>(&peakfile_dir_or_msp)->required(),
             "Input MSP file, with ID fields corresponding to id fields in input_file ("
             "the MSP filename not including the .msp extension) OR "
             "Directory containing files with spectra. Each file should be called <id>.txt, "
             "where <id> is the id specified in the input file, and contains a list of peaks "
             "'mass intensity' on each line, with either 'low','med' and 'high' lines beginning "
             "spectra of different energy levels, or 'energy0', 'energy1', "
             "etc. e.g:energy0 65.02 40.086.11 60.0 energy1 65.02 100.0 .")
            ("group,g", po::value<int>(&min_group)->default_value(-1),
                    "validation group for cross validation, if not provide cross validation"
                    "will apply on every group (n-fold cross validation )")
            ("num_fold,n", po::value<int>(&max_group)->default_value(0),
             "number for n-fold cross validation, if not provide, noy cross validation will be applied. "
             "If group number is set, this number will be igroned.")
            ("tmp_data_folder,t", po::value<std::string>(&data_folder)->default_value("tmp_data"),
             "Name of folder to write tmp data for training. If not specified will write to tmp_data")
            ("log_file,l", po::value<std::string>(&status_filename)->default_value("status.log"),
             "Name of file to write logging information as the program runs."
             "If not specified will write to status.log<group>, or status.log if no group is specified")
            ("start_energy,e", po::value<int>(&start_energy)->default_value(0),
             "Set to starting energy if want to start training part way through (single energy only -default 0)")
            ("start_repeat,r", po::value<int>(&start_repeat)->default_value(0),
             "Set to starting repeat if want to start training part way through (default 0)")
            ("fv_fragment_graphs_folder,a",  po::value<std::string>(&fv_fragment_graphs_folder)->default_value(""),
             "Name of folder to write and read fragement cache data for training. If not specified will write to "
             "tmp_data/fv_fragment_graphs_folder")
            ("no_train",po::value<bool>(&no_train)->default_value(false),"no training flag, defualt false");

    try {
        po::command_line_parser parser{argc, argv};
        parser.options(general_options).allow_unregistered().style(
                po::command_line_style::default_style |
                po::command_line_style::allow_slash_for_short);
        po::parsed_options parsed_options = parser.run();

        po::variables_map vm;

        store(parsed_options, vm);

        //help option
        if (vm.count("help")) {
            if (mpi_rank == MASTER)
                std::cout << general_options << std::endl;
            return 0;
        } else
            po::notify(vm);
    }
    catch (po::error &e) {
        if (mpi_rank == MASTER) {
            std::cerr << "ERROR: " << e.what() << std::endl << std::endl;
            std::cout << general_options << std::endl;
        }
        return 0;
    }

    //set group
    if (min_group != -1)
        max_group = min_group + 1;

    status_filename = data_folder + '/' + status_filename;
    std::string enumrated_output_folder = data_folder + "/enumerated_output";
    std::string predicted_output_folder = data_folder + "/predicted_output";
    if(fv_fragment_graphs_folder.empty())
        fv_fragment_graphs_folder = data_folder + "/fv_fragment_graphs";

    if (mpi_rank == MASTER) {

        //Create the tmp_data directory if it doesn't exist
        if (!boost::filesystem::exists(data_folder))
            boost::filesystem::create_directory(data_folder);
        if (!boost::filesystem::exists(enumrated_output_folder))
            boost::filesystem::create_directory(enumrated_output_folder);
        if (!boost::filesystem::exists(predicted_output_folder))
            boost::filesystem::create_directory(predicted_output_folder);

        boost::filesystem::path fv_fragment_graphs_folder_path(fv_fragment_graphs_folder);
        if (!boost::filesystem::exists(fv_fragment_graphs_folder_path))
            boost::filesystem::create_directories(fv_fragment_graphs_folder_path);

        //Delete the status file if it already exists
        if (boost::filesystem::exists(status_filename))
            boost::filesystem::remove_all(status_filename);
    }
    MPI_Barrier(MPI_COMM_WORLD);

    if (mpi_rank == MASTER) std::cout << "Initialising Feature Calculator..";
    FeatureCalculator fc(feature_filename);
    if (mpi_rank == MASTER) std::cout << "Done" << std::endl;

    if (mpi_rank == MASTER) std::cout << "Initialising Parameter Configuration..";
    config_t cfg;
    initConfig(cfg, config_filename, mpi_rank == MASTER);
    if (mpi_rank == MASTER) std::cout << "Done" << std::endl;

    if (mpi_rank == MASTER) std::cout << "Parsing input file...";
    std::vector<MolData> data;
    parseInputFile(data, input_filename, mpi_rank, mpi_nump, &cfg);
    if (mpi_rank == MASTER) std::cout << "Done" << std::endl;

    //Fragment Graph Computation (or load from file)
    // ═══════════════════════════════════════════════════════════════
    // OpenMP PARALLELIZED: each molecule's graph is independent
    // ═══════════════════════════════════════════════════════════════
    time_t before_fg, after_fg;
    before_fg = time(nullptr);
    int total_count = (int)data.size();
    if (mpi_rank == MASTER) {
        std::cout << "[Graph] Computing fragmentation graphs for " << total_count << " molecules..";
#ifdef _OPENMP
        std::cout << " [OpenMP: " << omp_get_max_threads() << " threads]";
#endif
        std::cout << std::endl;
    }

    // Pre-check which molecules have cached graphs
    // (file existence check is fast, do it sequentially to build work list)
    std::vector<bool> has_cache(total_count, false);
    std::vector<std::string> fv_filenames(total_count);
    for (int i = 0; i < total_count; i++) {
        fv_filenames[i] = fv_fragment_graphs_folder + "/" +
                          boost::lexical_cast<std::string>(data[i].getId()) + "_graph.fg";
        has_cache[i] = boost::filesystem::exists(fv_filenames[i]);
    }

    // Track results per molecule (thread-safe: each index written by one thread)
    std::vector<int> result_status(total_count, 0);  // 0=pending, 1=success, -1=exception
    std::vector<std::string> result_messages(total_count);
    std::vector<int> failed_indices;  // indices to remove after parallel section

    int progress_done = 0;
#ifdef _OPENMP
    #pragma omp parallel for schedule(dynamic)
#endif
    for (int i = 0; i < total_count; i++) {
        if (!no_train) {
            try {
                if (has_cache[i]) {
                    // Load from cached file (thread-safe: each molecule independent)
                    std::ifstream fv_ifs(fv_filenames[i].c_str(),
                                         std::ifstream::in | std::ios::binary);
                    data[i].readInFVFragmentGraphFromStream(fv_ifs);
                    fv_ifs.close();

                    std::ostringstream msg;
                    msg << "ID: " << data[i].getId() << " is Loaded from cached file "
                        << " Num Frag = " << data[i].getNumFragments()
                        << " Num Trans = " << data[i].getNumTransitions();
                    result_messages[i] = msg.str();
                    result_status[i] = 1;

                } else {
                    // Compute fragment graph (CPU-intensive, benefits from parallelization)
                    time_t before = time(nullptr);
                    data[i].computeFragmentGraphAndReplaceMolsWithFVs(&fc, true);
                    time_t after = time(nullptr);

                    std::ostringstream msg;
                    msg << "ID: " << data[i].getId() << " is Done. Time Elaspsed = "
                        << (after - before) << " Seconds "
                        << " Num Frag = " << data[i].getNumFragments()
                        << " Num Trans = " << data[i].getNumTransitions();
                    result_messages[i] = msg.str();
                    result_status[i] = 1;

                    // Write cache file (thread-safe: unique filename per molecule)
                    if (!boost::filesystem::exists(fv_filenames[i])) {
                        std::string fv_tmp_filename = fv_filenames[i] + "_tmp_" + std::to_string(i);
                        std::ofstream fv_out(fv_tmp_filename.c_str(),
                                            std::ios::out | std::ios::binary);
                        data[i].writeFVFragmentGraphToStream(fv_out);
                        fv_out.close();
                        boost::filesystem::rename(fv_tmp_filename, fv_filenames[i]);
                    }
                }
            }
            catch (std::exception &e) {
                std::ostringstream msg;
                msg << "Exception occurred computing fragment graph for "
                    << data[i].getId() << std::endl
                    << data[i].getSmilesOrInchi() << std::endl
                    << e.what();
                result_messages[i] = msg.str();
                result_status[i] = -1;
            }
        } else {
            result_status[i] = 1;
        }

        // Progress tracking (thread-safe: atomic increment + critical output)
#ifdef _OPENMP
        {
            int local_done;
            #pragma omp atomic capture
            local_done = ++progress_done;
            {
                #pragma omp critical(graph_progress)
                {
                    std::cout << "[Graph] " << local_done << "/" << total_count
                              << " molecules processed" << std::endl;
                }
            }
        }
#else
        ++progress_done;
        std::cout << "[Graph] " << progress_done << "/" << total_count
                  << " molecules processed" << std::endl;
#endif
    }

    // Write all log messages sequentially (preserves order, thread-safe file I/O)
    int success_count = 0, except_count = 0;
    {
        std::ofstream eout(status_filename.c_str(),
                           std::fstream::out | std::fstream::app);
        for (int i = 0; i < total_count; i++) {
            if (!result_messages[i].empty()) {
                eout << result_messages[i] << std::endl;
            }
            if (result_status[i] == 1) {
                success_count++;
            } else if (result_status[i] == -1) {
                except_count++;
                eout << except_count << " exceptions, from "
                     << except_count + success_count << " total" << std::endl;
            }
        }
        eout.close();
    }

    // Remove failed molecules from data vector (must be done sequentially, reverse order)
    for (int i = total_count - 1; i >= 0; i--) {
        if (result_status[i] == -1) {
            data.erase(data.begin() + i);
        }
    }

    MPI_Barrier(MPI_COMM_WORLD);
    after_fg = time(nullptr);
    if (mpi_rank == MASTER)
        std::cout << "[Graph] Completed: " << success_count << " success, "
                  << except_count << " exceptions" << std::endl;

    std::cout << mpi_rank << ": " << success_count << " successfully computed. " << except_count << " exceptions."
              << std::endl;

    if (mpi_rank == MASTER)
        std::cout << "[Graph] Total time: "
                  << (after_fg - before_fg) << " seconds" << std::endl;


    //Loading input spectra
    MPI_Barrier(MPI_COMM_WORLD);
    if (mpi_rank == MASTER) std::cout << "[Spectra] Loading " << data.size() << " spectra files..." << std::endl;
    bool spectra_in_msp = false;
    std::string pre_id = "";
    if (peakfile_dir_or_msp.substr(peakfile_dir_or_msp.size() - 4, 4) == ".msp") {
        spectra_in_msp = true;
        //Allow splitting of input data into multiple msps by naming them
        //some_file_name<P>.msp, where <P> gets replaced with the processor rank that uses that file
        pre_id = peakfile_dir_or_msp.substr(0, peakfile_dir_or_msp.size() - 4);
        if (pre_id.substr(pre_id.size() - 3, 3) == "<P>") {
            peakfile_dir_or_msp.replace(peakfile_dir_or_msp.size() - 7, 3, boost::lexical_cast<std::string>(mpi_rank));
        }
    }

    //MSP Setup
    MspReader *msp = nullptr;
    std::ostream *out_enum_msp, *out_pred_msp;
    std::ofstream of_emsp, of_pmsp;
    //Create the MSP lookup
    if (spectra_in_msp)
        msp = new MspReader(peakfile_dir_or_msp.c_str(), "");

    int spectra_loaded = 0;
    int spectra_total = (int)data.size();
    for (auto & mit : data ){
        if (!no_train) {
            if (spectra_in_msp)
                mit.readInSpectraFromMSP(*msp);
            else {
                std::string spec_file = peakfile_dir_or_msp + "/" + mit.getId() + ".txt";
                mit.readInSpectraFromFile(spec_file);
            }

            auto remove_msg = mit.removePeaksWithNoFragment(cfg.abs_mass_tol, cfg.ppm_mass_tol);
            std::ofstream eout;
            eout.open(status_filename.c_str(), std::fstream::out | std::fstream::app);
            eout << "ID: " << mit.getId() << " Remove Peaks With No Fragment. " << remove_msg << std::endl;
            eout.close();

            spectra_loaded++;
            if (mpi_rank == MASTER && (spectra_loaded % 20 == 0 || spectra_loaded == spectra_total)) {
                std::cout << "[Spectra] " << spectra_loaded << "/" << spectra_total
                          << " loaded" << std::endl;
            }
        }
    }

    MPI_Barrier(MPI_COMM_WORLD);
    if (mpi_rank == MASTER) std::cout << "[Spectra] All spectra loaded" << std::endl;

    //Training
    for (int group = min_group; group < max_group; group++) {
        if (mpi_rank == MASTER) std::cout << "Running EM to train parameters for Group " << group << std::endl;

        time_t before, after;
        before = time(nullptr);
        std::string param_filename = data_folder + "/param_output";
        param_filename += boost::lexical_cast<std::string>(group);
        param_filename += ".log";

        if (!no_train)
            trainSingleEnergyCFM(param_filename, cfg, fc, status_filename, group, data, start_energy, no_train,
                                 start_repeat);

        MPI_Barrier(MPI_COMM_WORLD);    //Wait for all threads
        if (mpi_rank == MASTER) std::cout << "Done" << std::endl;

        if (mpi_rank == MASTER) {
            after = time(nullptr);
            std::cout << "EM: Time Elaspsed = " << (after - before) << " Seconds" << std::endl;
        }

        //Open the output MSP files
        if (spectra_in_msp) {
            std::string proc_group_str =
                    "P" + boost::lexical_cast<std::string>(mpi_rank) + "G" + boost::lexical_cast<std::string>(group);
            std::string enum_msp_filename = data_folder + "/enumerated_output/especs_" + proc_group_str + ".msp";
            std::string pred_msp_filename = data_folder + "/predicted_output/pspecs_" + proc_group_str + ".msp";
            of_emsp.open(enum_msp_filename.c_str());
            of_pmsp.open(pred_msp_filename.c_str());
            if (!of_emsp.is_open() || !of_pmsp.is_open()) {
                std::cout << "Warning: Trouble opening msp output files" << std::endl;
                exit(1);
            }
            std::streambuf *ebuf = of_emsp.rdbuf(), *pbuf = of_pmsp.rdbuf();
            out_enum_msp = new std::ostream(ebuf);
            out_pred_msp = new std::ostream(pbuf);
        }

        if (mpi_rank == MASTER) std::cout << "Generating Peak Predictions for Group " << group << "..." << std::endl;
        Param *param;
        if (cfg.theta_function == NEURAL_NET_THETA_FUNCTION)
            param = new NNParam(param_filename);
        else
            param = new Param(param_filename);
        for (auto mit = data.begin(); mit != data.end(); ++mit) {
            if (mit->getGroup() != group) continue;
            if (!mit->hasComputedGraph()) continue;    //If we couldn't compute it's graph for some reason..

            //Predicted spectrum
            for (auto e = start_energy; e < cfg.spectrum_depths.size(); ++e)
                mit->computePredictedSpectra(*param, false, e, 1, 30, 100, cfg.use_log_scale_peak);

            if (spectra_in_msp)
                mit->writePredictedSpectraToMspFileStream(*out_pred_msp);
            else {
                std::string spectra_filename = data_folder + "/predicted_output/" + mit->getId() + ".log";
                mit->writePredictedSpectraToFile(spectra_filename);
            }
        }
        delete param;
        if (mpi_rank == MASTER) std::cout << "Done" << std::endl;

        if (spectra_in_msp) {
            of_emsp.close();
            of_pmsp.close();
            delete out_pred_msp;
            delete out_enum_msp;
        }

        MPI_Barrier(MPI_COMM_WORLD);    //Wait for all threads
    }

    if (spectra_in_msp) delete msp;


    MPI_Barrier(MPI_COMM_WORLD);    //Wait for all threads
    MPI_Finalize();

    return (0);
}

void
trainSingleEnergyCFM(std::string &param_filename, config_t &cfg, FeatureCalculator &fc, std::string &status_filename,
                     int group, std::vector<MolData> &data, int start_energy, int no_train, int start_repeat) {

    int mpi_rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &mpi_rank);
    Param *final_params;

    for (int energy = 0; energy < cfg.spectrum_depths.size(); energy++) {

        std::string eparam_filename = param_filename + boost::lexical_cast<std::string>(energy) + "e";
        std::string prev_eparam_filename = param_filename + boost::lexical_cast<std::string>(energy - 1) + "e";

        config_t se_cfg;
        initSingleEnergyConfig(se_cfg, cfg, energy);

        if (mpi_rank == MASTER) {
            std::cout << "[Training] ================================================" << std::endl;
            std::cout << "[Training] Starting Energy " << energy
                      << " (" << (energy+1) << "/" << cfg.spectrum_depths.size() << ")" << std::endl;
            std::cout << "[Training] ================================================" << std::endl;
        }

        if (energy >= start_energy && !no_train) {
            //Run EM multiple times with random restarts, taking the final one with the best Q
            double prev_Q = -DBL_MAX;
            if (energy > start_energy) start_repeat = 0;
            if (start_repeat > 0) { //Too messy to compute previous best Q...just print a warning
                std::cout
                        << "Warning: best Q for previous repeats unknown, may not pick up best params - check manually!"
                        << std::endl;
            }
            for (int repeat = start_repeat; repeat < se_cfg.num_em_restarts; repeat++) {

                EmModel *em;
                std::string repeat_filename = eparam_filename + boost::lexical_cast<std::string>(repeat);
                std::string out_filename = repeat_filename;
                if (!boost::filesystem::exists(repeat_filename)) repeat_filename = "";
                else if (energy > 0 && cfg.use_lower_energy_params_for_init)
                    repeat_filename = prev_eparam_filename + boost::lexical_cast<std::string>(repeat);
                if (cfg.theta_function == NEURAL_NET_THETA_FUNCTION)
                    em = new EmNNModel(&se_cfg, &fc, status_filename, repeat_filename);
                else
                    em = new EmModel(&se_cfg, &fc, status_filename, repeat_filename);

                double Q = em->trainModel(data, group, out_filename, energy);
                if (Q > prev_Q) {
                    if (mpi_rank == MASTER) {
                        std::cout << "Found better Q!" << std::endl;
                        em->writeParamsToFile(eparam_filename);
                    }
                    prev_Q = Q;
                }
                delete em;
            }
        }
        MPI_Barrier(MPI_COMM_WORLD);
        if (mpi_rank == MASTER)
            std::cout << "[Training] Energy " << energy << " completed" << std::endl;
        if (energy == start_energy) {
            if (cfg.theta_function == NEURAL_NET_THETA_FUNCTION)
                final_params = new NNParam(eparam_filename);
            else
                final_params = new Param(eparam_filename);
        } else if (energy > start_energy) {
            Param *eparam;
            if (cfg.theta_function == NEURAL_NET_THETA_FUNCTION)
                eparam = new NNParam(eparam_filename);
            else
                eparam = new Param(eparam_filename);
            final_params->appendNextEnergyParams(*eparam, energy);
            delete eparam;
        }
    }
    if (mpi_rank == MASTER) final_params->saveToFile(param_filename);
    delete final_params;
}


void
parseInputFile(std::vector<MolData> &data, std::string &input_filename, int mpi_rank, int mpi_nump, config_t *cfg) {

    std::string line, smiles_or_inchi, id;
    std::ifstream ifs(input_filename.c_str(), std::ifstream::in);
    int group, num_mols = 0;

    //Get the first line - the number of input molecules
    if (ifs.good()) {
        getline(ifs, line);
        num_mols = atoi(line.c_str());
    } else {
        std::cout << "Could not open input file " << input_filename << std::endl;
    }

    //Now get all the molecules
    int i = 0;
    while (ifs.good() && i < num_mols) {
        i++;

        getline(ifs, line);
        if (line.size() < 3) continue;

        std::stringstream ss(line);
        ss >> id >> smiles_or_inchi >> group;

        //Split the data between processors. Only load in data for this
        //processor
        if ((i % mpi_nump) == mpi_rank)
            data.push_back(MolData(id, smiles_or_inchi, group, cfg));
    }

}
