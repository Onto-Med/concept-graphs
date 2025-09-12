import spacy
from transformers import GPT2Tokenizer

models = {
    "spacy": (spacy.cli.download, ["de_core_news_sm", "de_dep_news_trf"],),
    "transformers": (GPT2Tokenizer.from_pretrained, ["gpt2", "dbmdz/german-gpt2"], {"force_download": True},)
}

if __name__ == '__main__':
    for k, v in models.items():
        _loader = v[0]
        for model in v[1]:
            if len(v) >= 3:
                _loader(model, **v[2])
            else:
                _loader(model)