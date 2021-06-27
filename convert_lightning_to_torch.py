from train import absTtilePL
from transformers import T5ForConditionalGeneration, T5Config

checkpoint_path = "ckpts/gpu4_bs32_e2.ckpt"
output_path = "ckpts/gpu4_bs32_e2"

config = T5Config.from_pretrained("t5-large", cache_dir=".cache", use_cache=True)
t5 = T5ForConditionalGeneration.from_pretrained("t5-large", cache_dir=".cache", config=config)
model = absTtilePL.load_from_checkpoint(checkpoint_path=checkpoint_path, model=t5)
model.model.save_pretrained(output_path)