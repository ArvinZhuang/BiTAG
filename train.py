from transformers import T5ForConditionalGeneration, T5Tokenizer, T5Config
import pytorch_lightning as pl
from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning import loggers as pl_loggers
import torch
from torch.utils.data import Dataset, DataLoader
import os
import argparse


class CheckpointEveryEpoch(pl.Callback):
    """
    Save a checkpoint every N steps, instead of Lightning's default that checkpoints
    based on validation loss.
    """

    def __init__(
        self,
        file_path,
    ):
        self.file_path = file_path

    def on_epoch_end(self, trainer: pl.Trainer, _):
        """ Check if we should save a checkpoint after every train epoch """
        ckpt_path = f"{self.file_path}_e{trainer.current_epoch}.ckpt"
        trainer.save_checkpoint(ckpt_path)


class dataset(Dataset):
    def __init__(self, path):
        self.titles = []
        self.abstracts = []
        files = os.listdir(path)
        for file in files:
            with open(path+file, 'r') as f:
                contents = f.readlines()

            for line in contents:
                title, abstract = line.strip().split('\t')
                if len(abstract) <= 50:
                    continue
                self.titles.append(title)
                self.abstracts.append(abstract)

    def __getitem__(self, index):
        title = self.titles[index]
        abstract = self.abstracts[index]
        return title, abstract

    def __len__(self):
        return len(self.titles)


class DataCollator(object):
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer

    def __call__(self, batch):
        abs_to_title_inputs = []
        abs_to_title_target = []
        title_to_abs_inputs = []
        title_to_abs_target = []
        for title, abstract in batch:
            abs_to_title_inputs.append("abstract: " + abstract)  # we use "abstract: " as prompt for abs_to_title task.
            abs_to_title_target.append(title)

            title_to_abs_inputs.append("title: " + title)  # we use "title: " as prompt for title_to_abs task.
            title_to_abs_target.append(abstract)

        abs_to_title_inputs = self.tokenizer(abs_to_title_inputs, return_tensors='pt', padding=True, truncation=True)
        abs_to_title_labels = self.tokenizer(abs_to_title_target, return_tensors='pt', padding=True, truncation=True).input_ids

        title_to_abs_inputs = self.tokenizer(title_to_abs_inputs, return_tensors='pt', padding=True, truncation=True)
        title_to_abs_labels = self.tokenizer(title_to_abs_target, return_tensors='pt', padding=True, truncation=True).input_ids

        return abs_to_title_inputs, abs_to_title_labels, title_to_abs_inputs, title_to_abs_labels


class BiTAG_pl(pl.LightningModule):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, inputs):
        outputs = self.model(**inputs, return_dict=True)
        return outputs

    def training_step(self, batch, batch_idx):
        abs_to_title_inputs, abs_to_title_labels, title_to_abs_inputs, title_to_abs_labels = batch

        abs_to_title_loss = self.model(**abs_to_title_inputs,
                                       labels=abs_to_title_labels).loss

        title_to_abs_loss = self.model(**title_to_abs_inputs,
                                       labels=title_to_abs_labels).loss

        loss = (abs_to_title_loss + title_to_abs_loss)/2
        self.log("loss", loss)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=2e-5)
        return optimizer


if __name__ == "__main__":
    seed_everything(313)
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_model", type=str, default="t5-large")
    parser.add_argument("--output_path", type=str, default="ckpts/BiTAG")

    # training config
    parser.add_argument("--num_gpus", type=int, default=4)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--num_nodes", type=int, default=1, help="For multi-node multi-gpu training.")
    parser.add_argument("--num_workers", type=int, default=8, help="Number of works for dataloader.")
    parser.add_argument("--max_epochs", type=int, default=10, help="The maximum training epochs.")
    args = parser.parse_args()

    config = T5Config.from_pretrained(args.base_model,
                                      cache_dir=".cache",
                                      use_cache=False,
                                      gradient_checkpointing=True,  # trade-off GPU training speed and memory.
                                      )
    model = T5ForConditionalGeneration.from_pretrained(args.base_model, cache_dir=".cache", config=config)
    tokenizer = T5Tokenizer.from_pretrained(args.base_model, cache_dir=".cache")
    dataset = dataset('data/')
    tb_logger = pl_loggers.TensorBoardLogger('logs/')

    loader = DataLoader(dataset=dataset,
                        batch_size=args.batch_size,
                        drop_last=True,
                        pin_memory=True,
                        shuffle=True,
                        num_workers=args.num_workers,
                        collate_fn=DataCollator(tokenizer))

    trainer = Trainer(max_epochs=args.max_epochs,
                      gpus=args.num_gpus,
                      num_nodes=args.num_nodes,
                      checkpoint_callback=False,
                      logger=tb_logger,
                      accelerator="ddp",
                      plugins='ddp_sharded',
                      log_every_n_steps=10,
                      callbacks=[CheckpointEveryEpoch(args.output_path)]
                      )
    trainer.fit(BiTAG_pl(model), loader)
