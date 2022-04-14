import torch
import os
import wandb
import pickle
import random
import copy
import numpy as np
from datetime import datetime
from tqdm import tqdm

# from ..lib import HVO_Sequence


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class GrooveMidiDatasetADT(torch.utils.data.Dataset):
    def __init__(self, data=None, load_dataset_path=None, **kwargs):
        """
        Groove Midi Dataset Loader. Max number of items in dataset is N x M x K where N is the number of items in the
        subset, M the maximum number of soundfonts to sample from for each item (aug) and K is the maximum number
        of voice combinations.

        @param data:              GrooveMidiDataset subset generated with the Subset_Creator
        @param subset_info:         Dictionary with the routes and filters passed to the Subset_Creator to generate the
                                    subset. Example:
                                    subset_info = {
                                    "pickle_source_path": '../../processed_dataset/datasets_extracted_locally/GrooveMidi/hvo_0.4.2'
                                               '/Processed_On_17_05_2021_at_22_32_hrs',
                                    "subset": 'GrooveMIDI_processed_train',
                                    "metadata_csv_filename": 'metadata.csv',
                                    "hvo_pickle_filename": 'hvo_sequence_data.obj',
                                    "filters": "bpm": ["beat"],
                                    }

        @param max_seq_len:             Max_length of sequences
        @param mso_params:      Dictionary with the parameters for calculating the Multiband Synthesized Onsets.
                                    Refer to `hvo_sequence.hvo_seq.mso()` for the documentation
        @param sf_path:             Path with soundfonts
        @param aug:            Maximum number of soundfonts to sample from for each example
        @param dataset_name:        Dataset name (for experiment tracking)
        """
        self.__version__ = "0.0.0"
        # pickle file module not found fix
        self.__module__ = "dataset"

        # get params
        if load_dataset_path:
            self.dataset_name = (
                load_dataset_path.split("/")[-1]
                if load_dataset_path.split("/")[-1]
                else load_dataset_path.split("/")[-2]
            )
            self.load_params_from_pickle(load_dataset_path)
        else:
            # default values for kwargs
            self.max_seq_len = kwargs.get("max_seq_len", 32)
            self.mso_params = kwargs.get(
                "mso_params",
                {
                    "sr": 44100,
                    "n_fft": 1024,
                    "win_length": 1024,
                    "hop_length": 441,
                    "n_bins_per_octave": 16,
                    "n_octaves": 9,
                    "f_min": 40,
                    "mean_filter_size": 22,
                },
            )

            self.sf_path = kwargs.get("sf_path", "soundfonts/")
            self.aug_coefficient = kwargs.get("aug_coefficient", None)
            self.timestamp = datetime.now().strftime("%d_%m_%Y_at_%H_%M_hrs")
            self.dataset_name = (
                "Dataset_" + self.timestamp
                if kwargs.get("dataset_name") is None
                else kwargs.get("dataset_name", "Dataset_" + self.timestamp)
            )
            self.subset_info = kwargs.get(
                "subset_info",
                {
                    "pickle_source_path": "",
                    "subset": "",
                    "metadata_csv_filename": "",
                    "hvo_pickle_filename": "",
                    "filters": "",
                },
            )
            self.split = kwargs.get("split", "")

            self.sfs_list = get_sf_list(self.sf_path)

            if self.aug_coefficient is not None:
                assert self.aug_coefficient <= len(
                    self.sfs_list
                ), "aug can not be larger than number of available soundfonts"

            self.save_dataset_path = kwargs.get(
                "save_dataset_path", os.path.join("dataset", self.dataset_name)
            )
        # process dataset
        print("GMD path: ", self.subset_info["pickle_source_path"])
        processed_dataset = (
            self.load_dataset_from_pickle(load_dataset_path)
            if load_dataset_path
            else self.process_dataset(data)
        )

        # store processed dataset in dataset attrs
        for key in processed_dataset.keys():
            self.__setattr__(key, processed_dataset[key])

        # dataset params dict
        params = self.get_params()

        # log params to wandb
        if wandb.ensure_configured():  # if running experiment file with wandb.init()
            wandb.config.update(params, allow_val_change=True)  # update defaults

        # save dataset to pickle file
        if load_dataset_path is None:
            if not os.path.exists(self.save_dataset_path):
                os.makedirs(self.save_dataset_path)
            print(self.save_dataset_path)
            # move tensor to cpu (tensors saved while on gpu cannot be loaded from pickle file in cpu)
            processed_dataset["processed_inputs"] = processed_dataset[
                "processed_inputs"
            ].to(device="cpu")
            processed_dataset["processed_outputs"] = processed_dataset[
                "processed_outputs"
            ].to(device="cpu")

            # save to pickle
            params_pickle_filename = os.path.join(
                self.save_dataset_path,
                "{}_{}_{}_params.pickle".format(
                    self.dataset_name, self.split, self.__version__
                ),
            )
            dataset_pickle_filename = os.path.join(
                self.save_dataset_path,
                "{}_{}_{}_dataset.pickle".format(
                    self.dataset_name, self.split, self.__version__
                ),
            )
            save_to_pickle(params, params_pickle_filename)
            save_to_pickle(processed_dataset, dataset_pickle_filename)

            print("Saved dataset to path: ", self.save_dataset_path)

    def process_dataset(self, data):
        self.save_dataset_path = os.path.join(
            os.path.join(self.save_dataset_path, self.__version__), self.split
        )

        print("GrooveMidiDatasetADT version " + self.__version__)

        # init lists to store hvo sequences and processed io
        hvo_sequences, hvo_sequences_outputs = [], []
        processed_inputs, processed_outputs = [], []

        # init list with configurations
        hvo_index, soundfonts = [], []

        for hvo_idx, hvo_seq in enumerate(
            tqdm(data, desc="processing dataset {}".format(self.subset_info["subset"]))
        ):

            all_zeros = not np.any(hvo_seq.hvo.flatten())  # silent patterns

            if (
                len(hvo_seq.time_signatures) == 1 and not all_zeros
            ):  # ignore if time_signature change happens

                # pad with zeros to match max_len
                hvo_seq = pad_to_match_max_seq_len(hvo_seq, self.max_seq_len)

                # append hvo_seq to hvo_sequences list
                hvo_sequences.append(hvo_seq)

                # get voices and sf combinations
                sfs = random.choices(self.sfs_list, k=self.aug_coefficient)
                # for every sf and voice combination
                for sf in sfs:

                    # store hvo and sf
                    hvo_index.append(hvo_idx)
                    soundfonts.append(sf)
                    hvo_sequences_outputs.append(hvo_seq)

                    # processed inputs: mso
                    mso = hvo_seq.mso(sf_path=sf, **self.mso_params)
                    processed_inputs.append(mso)

                    # processed outputs: hvo_seq
                    processed_outputs.append(hvo_seq.hvo)

        # convert inputs and outputs to torch tensors
        processed_inputs = torch.Tensor(processed_inputs).to(device=device)
        processed_outputs = torch.Tensor(processed_outputs).to(device=device)

        processed_dict = {
            "processed_inputs": processed_inputs,
            "processed_outputs": processed_outputs,
            "hvo_sequences": hvo_sequences,
            "hvo_sequences_outputs": hvo_sequences_outputs,
            "hvo_index": hvo_index,
            "soundfonts": soundfonts,
        }

        return processed_dict

    # load from pickle

    def load_params_from_pickle(self, dataset_path):
        params_file = os.path.join(
            dataset_path,
            list(
                filter(
                    lambda x: x.endswith("_params.pickle"),
                    os.listdir(dataset_path),
                )
            )[0],
        )

        with open(params_file, "rb") as f:
            params = pickle.load(f)

        for key in params.keys():
            self.__setattr__(key, params[key])

        print("Loaded parameters from path: ", params_file)

    def load_dataset_from_pickle(self, dataset_path):
        pickle_file = os.path.join(
            dataset_path,
            list(
                filter(
                    lambda x: x.endswith("_dataset.pickle"),
                    os.listdir(dataset_path),
                )
            )[0],
        )

        with open(pickle_file, "rb") as f:
            processed_dataset = pickle.load(f)

        for key in processed_dataset.keys():
            self.__setattr__(key, processed_dataset[key])

        print("Loaded dataset from path: ", pickle_file)

        print(str(self.__len__()) + " items")

        return processed_dataset

    # getters

    def get_hvo_sequence(self, idx):
        hvo_idx = self.hvo_index[idx]
        return self.hvo_sequences[hvo_idx]

    def get_soundfont(self, idx):
        return self.soundfonts[idx]

    def get_params(self):
        params = copy.deepcopy(self.__dict__)

        # delete dataset attr
        params["processed_inputs"] = {}
        params["processed_outputs"] = {}
        params["hvo_sequences"] = {}

        del params["processed_inputs"]
        del params["processed_outputs"]
        del params["hvo_sequences"]

        return params

    # dataset methods

    def __len__(self):
        return len(self.processed_inputs)

    def __getitem__(self, idx):
        return self.processed_inputs[idx], self.processed_outputs[idx], idx


class GrooveMidiDatasetVAE(GrooveMidiDatasetADT):
    def __init__(self, data=None, load_dataset_path=None, **kwargs):

        super(GrooveMidiDatasetVAE, self).__init__(
            data=data, load_dataset_path=load_dataset_path, **kwargs
        )

        # audio attrs inherited from GMDInfilling
        del self.mso_params
        del self.sfs_list
        del self.sf_path
        del self.aug_coefficient

    # override preprocessing dataset method
    def process_dataset(self, data):
        self.__version__ = "0.0.0"
        self.save_dataset_path = os.path.join(
            os.path.join(self.save_dataset_path, self.__version__), self.split
        )
        print("GrooveMidiDatasetVAE version " + self.__version__)

        # init lists to store hvo sequences and processed io
        hvo_sequences = []
        hvo_sequences_inputs, hvo_sequences_outputs = [], []
        processed_inputs, processed_outputs = [], []

        # init list with configurations
        hvo_index = []

        for hvo_idx, hvo_seq in enumerate(
            tqdm(
                data, desc="Preprocessing dataset {}".format(self.subset_info["subset"])
            )
        ):

            all_zeros = not np.any(hvo_seq.hvo.flatten())  # silent patterns

            if (
                len(hvo_seq.time_signatures) == 1 and not all_zeros
            ):  # ignore if time_signature change happens
                # add metadata to hvo_seq scores
                # add_metadata_to_hvo_seq(hvo_seq, hvo_idx, self.metadata)

                # pad with zeros to match max_len
                hvo_seq = pad_to_match_max_seq_len(hvo_seq, self.max_seq_len)

                # append hvo_seq to hvo_sequences list
                hvo_index.append(hvo_idx)
                hvo_sequences.append(hvo_seq)

                # inputs and outputs are the same
                hvo_sequences_outputs.append(hvo_seq)

                # processed inputs
                processed_inputs.append(hvo_seq.hvo)

                # processed outputs
                processed_outputs.append(hvo_seq.hvo)

        # convert inputs and outputs to torch tensors
        processed_inputs = torch.Tensor(processed_inputs).to(device=device)
        processed_outputs = torch.Tensor(processed_outputs).to(device=device)

        processed_dict = {
            "processed_inputs": processed_inputs,
            "processed_outputs": processed_outputs,
            "hvo_sequences": hvo_sequences,
            "hvo_sequences_outputs": hvo_sequences_outputs,
            "hvo_index": hvo_index,
        }

        return processed_dict


# utils


def get_sf_list(sf_path):
    if not isinstance(sf_path, list) and sf_path.endswith(
        ".sf2"
    ):  # if only one sf is given
        sfs_list = [sf_path]
    elif not isinstance(sf_path, list) and os.path.isdir(
        sf_path
    ):  # if dir with sfs is given
        sfs_list = [
            os.path.join(sf_path) + sf
            for sf in os.listdir(sf_path)
            if sf.endswith(".sf2")
        ]
    else:
        sfs_list = sf_path  # list of paths
    return sfs_list


def pad_to_match_max_seq_len(hvo_seq, max_len):
    pad_count = max(max_len - hvo_seq.hvo.shape[0], 0)
    hvo_seq.hvo = np.pad(hvo_seq.hvo, ((0, pad_count), (0, 0)), "constant")
    hvo_seq.hvo = hvo_seq.hvo[:max_len, :]  # in case seq exceeds max len

    return hvo_seq


def save_to_pickle(obj, filename):
    with open(filename, "wb") as f:
        pickle.dump(obj, f)
