from train import absTtilePL

checkpoint_path = "ckpts/gpu4_bs32_e2.ckpt"
output_path = "ckpts/gpu4_bs32_e2"

model = absTtilePL.load_from_checkpoint(checkpoint_path=checkpoint_path)
model.model.save_pretrained(output_path)