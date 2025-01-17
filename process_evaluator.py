import torch
from evaluator import ADTEvaluator
from process_dataset import load_processed_dataset

params = {
    "dataset_paths": {
        "GrooveADT_testing": {
            "train": "datasets/GrooveADT_testing/0.0.0/train",
            "test": "datasets/GrooveADT_testing/0.0.0/test",
            "validation": "datasets/GrooveADT_testing/0.0.0/validation",
        },
        "GrooveADT": {
            "train": "datasets/GrooveADT/0.0.0/train",
            "test": "datasets/GrooveADT/0.0.0/test",
            "validation": "datasets/GrooveADT/0.0.0/validation",
        },
        "GrooveVAE_testing": {
            "train": "datasets/GrooveVAE_testing/0.0.0/train",
            "test": "datasets/GrooveVAE_testing/0.0.0/test",
            "validation": "datasets/GrooveVAE_testing/0.0.0/validation",
        },
        "GrooveVAE": {
            "train": "datasets/GrooveVAE/0.0.0/train",
            "test": "datasets/GrooveVAE/0.0.0/test",
            "validation": "datasets/GrooveVAE/0.0.0/validation",
        },
    },
    "evaluator": {
        "n_samples_to_use": 1681,  # 2048
        "n_samples_to_synthesize_visualize_per_subset": 10,  # 10
        "save_evaluator_path": "evaluators/",
    },
}

if __name__ == "__main__":

    testing = False

    exps = ["GrooveVAE", "GrooveVAE_testing"]
    splits = ["test", "train", "validation"]
    for exp in exps:
        print("------------------------\n" + exp + "\n------------------------\n")
        for split in splits:
            print("Split: ", split)

            _exp = exp + "_testing" if testing else exp
            if testing:
                params["evaluator"]["n_samples_to_use"] = 10
                params["evaluator"]["n_samples_to_synthesize_visualize_per_subset"] = 5

            dataset = load_processed_dataset(params["dataset_paths"][_exp][split], _exp)

            evaluator = ADTEvaluator(
                pickle_source_path=dataset.subset_info["pickle_source_path"],
                set_subfolder=dataset.subset_info["subset"],
                hvo_pickle_filename=dataset.subset_info["hvo_pickle_filename"],
                max_hvo_shape=(32, 27),
                n_samples_to_use=params["evaluator"]["n_samples_to_use"],
                n_samples_to_synthesize_visualize_per_subset=params["evaluator"][
                    "n_samples_to_synthesize_visualize_per_subset"
                ],
                _identifier=split.capitalize() + "_Set",
                disable_tqdm=False,
                analyze_heatmap=True,
                analyze_global_features=False,  # pdf
                dataset=dataset,
                device="cuda" if torch.cuda.is_available() else "cpu",
            )

            evaluator.save_as_pickle(
                save_evaluator_path=params["evaluator"]["save_evaluator_path"]
            )
