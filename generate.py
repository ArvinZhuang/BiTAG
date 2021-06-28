from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
import yaml
import torch
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

if __name__ == "__main__":
    with open("generation_config.yaml") as f:
        config = yaml.load(f.read(), Loader=yaml.FullLoader)

    model_name_or_path = config["model_name_or_path"]
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name_or_path, cache_dir=".cache").to(DEVICE)
    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, cache_dir=".cache")

    text = ""
    if config["task"] == "title_to_abs":
        text = "title: " + config["text"]
    elif config["task"] == "abs_to_title":
        text = "abstract: " + config["text"]
    else:
        raise Exception("'task' has to be 'title_to_abs' or 'abs_to_title'.")

    print(text)
    input_ids = tokenizer.encode(text, return_tensors='pt').to(DEVICE)

    if config["search_method"] == "beam":
        outputs = model.generate(
            input_ids,
            max_length=config["beam_search_config"]["max_length"],
            num_beams=config["beam_search_config"]["num_beams"],
            num_return_sequences=config["beam_search_config"]["num_return_sequences"],
            early_stopping=True
        )
    elif config["search_method"] == "sampling":
        outputs = model.generate(
            input_ids,
            do_sample=True,
            max_length=config["sampling_search_config"]["max_length"],
            top_p=config["sampling_search_config"]["top_p"],
            top_k=config["sampling_search_config"]["top_k"],
            num_return_sequences=config["sampling_search_config"]["num_return_sequences"],
            temperature=float(config["sampling_search_config"]["temperature"])
        )
    else:
        raise Exception("'search_method' has to be 'sampling' or 'beam'.")

    print("Output:\n" + 100 * '-')
    for i, output in enumerate(outputs):
        print("{}: {}".format(i+1, tokenizer.decode(output, skip_special_tokens=True)))
