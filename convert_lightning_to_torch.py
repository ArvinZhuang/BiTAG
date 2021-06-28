from train import BiTAG_pl
from transformers import T5ForConditionalGeneration, T5Config, T5Tokenizer
import argparse


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_type", type=str, required=True)
    parser.add_argument("--model_path", type=str, required=True)
    args = parser.parse_args()

    tokenizer = T5Tokenizer.from_pretrained(args.model_type, cache_dir=".cache")
    tokenizer.save_pretrained(''.join(args.model_path.split('.')[:-1]))

    config = T5Config.from_pretrained(args.model_type, cache_dir=".cache", use_cache=True)
    t5 = T5ForConditionalGeneration.from_pretrained(args.model_type, cache_dir=".cache", config=config)
    pl_model = BiTAG_pl.load_from_checkpoint(checkpoint_path=args.model_path, model=t5)
    pl_model.model.save_pretrained(''.join(args.model_path.split('.')[:-1]))
