from transformers import T5ForConditionalGeneration, T5Tokenizer, T5Config
import json

if __name__ == "__main__":
    with open("generation_config.json") as f:
        config = json.loads(f.read())

    t5config = T5Config.from_pretrained("t5-large", cache_dir=".cache", use_cache=True)
    model_name = "ckpts/gpu4_bs32_e2"
    model = T5ForConditionalGeneration.from_pretrained(model_name, cache_dir=".cache", config=t5config)
    tokenizer = T5Tokenizer.from_pretrained("t5-large", cache_dir=".cache")

    text = ""
    if config["task"] == "title_to_abs":
        text = "title: " + config["text"]

    if config["task"] == "abs_to_title":
        text = "abstract: " + config["text"]
    print(text)
    input_ids = tokenizer.encode(text, return_tensors='pt')

    # beam_outputs = model.generate(
    #     input_ids,
    #     max_length=config["max_length"],
    #     num_beams=config["num_beams"],
    #     # no_repeat_ngram_size=2,
    #     num_return_sequences=config["num_return_sequences"],
    #     early_stopping=True
    # )
    #
    # print("Output:\n" + 100 * '-')
    # for i, beam_output in enumerate(beam_outputs):
    #     print("{}: {}".format(i, tokenizer.decode(beam_output, skip_special_tokens=True)))

    sample_outputs = model.generate(
        input_ids,
        do_sample=True,
        max_length=500000,
        top_p=0.95,
        top_k=50,
        num_return_sequences=10,
        temperature=1.5
    )
    print("Output:\n" + 100 * '-')
    for i, sample_output in enumerate(sample_outputs):
        print("{}: {}".format(i, tokenizer.decode(sample_output, skip_special_tokens=True)))
