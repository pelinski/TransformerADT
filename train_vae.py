import os
import torch
import wandb
from torch.utils.data import DataLoader
import yaml
import argparse
import pprint

from process_dataset import load_processed_dataset
from evaluator import init_evaluator, log_eval
from utils import eval_log_freq
from vae.train import initialize_model, calculate_loss, train_loop

parser = argparse.ArgumentParser()
parser.add_argument("--paths", help="paths file", default="configs/paths.yaml")
parser.add_argument("--testing", help="testing mode", default=False)
parser.add_argument("--wandb", help="log to wandb", default=True)
parser.add_argument("--eval_train", help="evaluator train set", default=True)
parser.add_argument("--eval_test", help="evaluator test set", default=True)
parser.add_argument("--eval_validation", help="evaluator validation set", default=True)
parser.add_argument(
    "--only_final_eval", help="only final total evaluation", default=False
)  # sweeps
parser.add_argument("--dump_eval", help="dump evaluator file", default=True)
parser.add_argument("--load_model", help="load model parameters", default=None)
parser.add_argument("--notes", help="wandb run notes", default=None)
parser.add_argument("--tags", help="wandb run tags", default=None)

# hyperparameters
parser.add_argument(
    "--config",
    help="yaml config file. if given, the rest of the arguments are not taken into "
    "account",
    default=None,
)
parser.add_argument("--experiment", help="experiment id", default=None)
parser.add_argument("--in_channels", help="input channels", default=1, type=int)
parser.add_argument("--latent_dim", help="latent dimension", default=32, type=int)
parser.add_argument("--out_x", help="output x dimension", default=32, type=int)
parser.add_argument("--out_y", help="output y dimension", default=32, type=int)
parser.add_argument(
    "--optimizer_algorithm", help="optimizer_algorithm", default="sgd", type=str
)

parser.add_argument(
    "--hit_loss_penalty",
    help="non_hit loss multiplier (between 0 and 1)",
    default=1,
    type=float,
)
parser.add_argument("--batch_size", help="batch size", default=16, type=int)
parser.add_argument(
    "--dim_feedforward", help="feed forward layer dimension", default=256, type=int
)
parser.add_argument("--learning_rate", help="learning rate", default=0.05, type=float)
parser.add_argument("--epochs", help="number of training epochs", default=100, type=int)

args = parser.parse_args()

# args are loaded all from config file or all from cli
if args.config is not None:
    with open(args.config, "r") as f:
        hyperparameters = yaml.safe_load(f)
else:
    hyperparameters = dict(
        optimizer_algorithm=args.optimizer_algorithm,
        in_channels=args.in_channels,
        latent_dim=args.latent_dim,
        out_x=args.out_x,
        out_y=args.out_y,
        hit_loss_penalty=args.hit_loss_penalty,
        batch_size=args.batch_size,
        dim_feedforward=args.dim_feedforward,
        learning_rate=args.learning_rate,
        epochs=args.epochs,
        load_model=args.load_model,
    )

if args.testing:
    hyperparameters["epochs"] = 1

# config files without experiment specified
if args.experiment is not None:
    hyperparameters["experiment"] = args.experiment

assert "experiment" in hyperparameters.keys(), "experiment not specified"

pprint.pprint(hyperparameters)

with open(args.paths, "r") as f:
    paths = yaml.safe_load(f)

os.environ["WANDB_MODE"] = "online" if args.wandb else "offline"

if __name__ == "__main__":
    wandb.init(
        config=hyperparameters,
        project=hyperparameters["experiment"],
        job_type="train",
        notes=args.notes,
        tags=args.tags,
        settings=wandb.Settings(start_method="fork"),
    )

    params = {
        "model": {
            "experiment": wandb.config.experiment,
            "optimizer": wandb.config.optimizer_algorithm,
            "in_channels": wandb.config.in_channels,
            "latent_dim": wandb.config.latent_dim,
            "out_x": wandb.config.out_x,
            "out_y": wandb.config.out_y,
            "max_len": 32,
            "device": "cuda" if torch.cuda.is_available() else "cpu",
        },
        "training": {
            "learning_rate": wandb.config.learning_rate,
            "batch_size": wandb.config.batch_size,
            "hit_loss_penalty": wandb.config.hit_loss_penalty
            #        'lr_scheduler_step_size': 30,
            #        'lr_scheduler_gamma': 0.1
        },
        "load_model": wandb.config.load_model,
    }

    # log params to wandb
    wandb.config.update(params["model"])

    # initialize model
    model, optimizer, initial_epoch = initialize_model(params)
    wandb.watch(model, log_freq=1000)

    # load dataset
    dataset_train = load_processed_dataset(
        paths[wandb.config.experiment]["datasets"]["train"], wandb.config.experiment
    )
    dataloader_train = DataLoader(
        dataset_train, batch_size=wandb.config.batch_size, shuffle=True, pin_memory=True
    )

    if args.eval_train:
        evaluator_train = init_evaluator(
            paths[wandb.config.experiment]["evaluators"]["train"],
            device=params["model"]["device"],disable_tqdm=True
        )
    if args.eval_test:
        evaluator_test = init_evaluator(
            paths[wandb.config.experiment]["evaluators"]["test"],
            device=params["model"]["device"],disable_tqdm=True
        )
    if args.eval_validation:
        evaluator_validation = init_evaluator(
            paths[wandb.config.experiment]["evaluators"]["validation"],
            device=params["model"]["device"],disable_tqdm=True
        )

    BCE_fn, MSE_fn = (
        torch.nn.BCELoss(reduction="none"),
        torch.nn.MSELoss(reduction="none"),
    )

    total_epochs = wandb.config.epochs
    epoch_save_all, epoch_save_partial = eval_log_freq(
        total_epochs=total_epochs,
        initial_epochs_lim=10,
        initial_step_partial=1,
        initial_step_all=1,
        secondary_step_partial=10,
        secondary_step_all=20,
        only_final=args.only_final_eval,
    )
    ep = initial_epoch
    for i in range(initial_epoch, total_epochs):

        print(f"Epoch {ep}\n-------------------------------")
        train_loop(
            dataloader=dataloader_train,
            groove_vae=model,
            opt=optimizer,
            epoch=ep,
            loss_fn=calculate_loss,
            bce_fn=BCE_fn,
            mse_fn=MSE_fn,
            device=params["model"]["device"],
            test_inputs=evaluator_test.processed_inputs if args.eval_test else None,
            test_gt=evaluator_test.processed_gt if args.eval_test else None,
            validation_inputs=evaluator_validation.processed_inputs
            if args.eval_validation
            else None,
            validation_gt=evaluator_validation.processed_gt
            if args.eval_validation
            else None,
            hit_loss_penalty=wandb.config.hit_loss_penalty,
            save=(ep in epoch_save_partial or ep in epoch_save_all),
        )
        print("-------------------------------\n")

        # if ep in epoch_save_partial or ep in epoch_save_all:
        if args.eval_train:
            # evaluator_train._identifier = 'Train_Set_Epoch_{}'.format(ep)
            evaluator_train._identifier = "Train_Set"
            log_eval(
                evaluator_train,
                model,
                log_media=ep in epoch_save_all,
                epoch=ep,
                dump=args.dump_eval,
            )

        if args.eval_test:
            # evaluator_test._identifier = 'Test_Set_Epoch_{}'.format(ep)
            evaluator_test._identifier = "Test_Set"
            log_eval(
                evaluator_test,
                model,
                log_media=ep in epoch_save_all,
                epoch=ep,
                dump=args.dump_eval,
            )

        if args.eval_validation:
            # evaluator_test._identifier = 'Validation_Set_Epoch_{}'.format(ep)
            evaluator_validation._identifier = "Validation_Set"
            log_eval(
                evaluator_validation,
                model,
                log_media=ep in epoch_save_all,
                epoch=ep,
                dump=args.dump_eval,
            )

        wandb.log({"epoch": ep}, commit=True)

        ep += 1

    wandb.finish()
