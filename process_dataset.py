import copy
from dataset import GrooveMidiDatasetADT, GrooveMidiDatasetVAE
from preprocessed_dataset.Subset_Creators.subsetters import GrooveMidiSubsetter

subset_info = {
    "pickle_source_path": "GMD2HVO_PreProcessing/processed_dataset/Processed_On_12_04_2022_at_18_47_hrs",
    "subset": "GrooveMIDI_processed_",
    "metadata_csv_filename": "metadata.csv",
    "hvo_pickle_filename": "hvo_sequence_data.obj",
    "filters": {"beat_type": ["beat"], "time_signature": ["4-4"]},
}
params = {
    "GrooveADT": {
        "dataset_name": "GrooveADT",
        "subset_info": subset_info,
        "mso_params": {
            "sr": 44100,
            "n_fft": 1024,
            "win_length": 1024,
            "hop_length": 441,
            "n_bins_per_octave": 16,
            "n_octaves": 9,
            "f_min": 40,
            "mean_filter_size": 22,
        },
        "sf_path": "soundfonts/Standard_Drum_Kit.sf2",
        "aug_coefficient": 1,
        "save_dataset_path": "datasets/GrooveADT/",
    },
    "GrooveVAE": {
        "dataset_name": "GrooveVAE",
        "subset_info": subset_info,
        "save_dataset_path": "datasets/GrooveVAE/",
    },
}


def process_dataset(params, exp):
    _, subset_list = GrooveMidiSubsetter(
        pickle_source_path=params["subset_info"]["pickle_source_path"],
        subset=params["subset_info"]["subset"],
        hvo_pickle_filename=params["subset_info"]["hvo_pickle_filename"],
        list_of_filter_dicts_for_subsets=[params["subset_info"]["filters"]],
    ).create_subsets()

    if exp.startswith("GrooveADT"):
        _dataset = GrooveMidiDatasetADT(data=subset_list[0], **params)
    elif exp.startswith("GrooveVAE"):
        _dataset = GrooveMidiDatasetVAE(data=subset_list[0], **params)
    else:
        return None
    return _dataset


def load_processed_dataset(load_dataset_path, exp):
    if exp.startswith("GrooveADT"):
        print("Loading GrooveMidiADT dataset..")
        _dataset = GrooveMidiDatasetADT(load_dataset_path=load_dataset_path)
    elif exp.startswith("GrooveVAE"):
        print("Loading GrooveMidiVAE dataset..")
        _dataset = GrooveMidiDatasetVAE(load_dataset_path=load_dataset_path)
    else:
        return None

    return _dataset


if __name__ == "__main__":

    testing = True

    exp = "GrooveVAE"

    # change experiment and split here
    splits = ["train", "test", "validation"]

    if testing:
        params[exp]["subset_info"]["filters"]["master_id"] = ["drummer2/session2/8"]
        params[exp]["dataset_name"] = params[exp]["dataset_name"] + "_testing"
        params[exp]["save_dataset_path"] = (
            "datasets/" + params[exp]["dataset_name"] + "/"
        )

    print(
        "------------------------\n"
        + params[exp]["dataset_name"]
        + "\n------------------------\n"
    )

    for split in splits:
        _params = copy.deepcopy(params[exp])
        _params["split"] = split
        _params["subset_info"]["subset"] = _params["subset_info"]["subset"] + split

        process_dataset(_params, exp)
