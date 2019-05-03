import torch

import encoder.utils as utils

# Load model
from models import InferSent

# Download nltk: punkt
import nltk
nltk.download('punkt')

model_version = 2
MODEL_PATH = "../encoder/infersent%s.pkl" % model_version
params_model = {'bsize': 64,
                'word_emb_dim': 300,
                'enc_lstm_dim': 2048,
                'pool_type': 'max',
                'dpout_model': 0.0,
                'version': model_version}

model = InferSent(params_model)

model.load_state_dict(torch.load(MODEL_PATH))

# Keep it on CPU or put it on GPU
use_cuda = False
model = model.cuda() if use_cuda else model

# If infersent1 -> use GloVe embeddings.
# If infersent2 -> use InferSent embeddings.
W2V_PATH = '../dataset/GloVe/glove.840B.300d.txt' if model_version == 1 \
    else '../dataset/fastText/crawl-300d-2M-subword.vec'
model.set_w2v_path(W2V_PATH)

# Load embeddings of K most frequent words
model.build_vocab_k_words(K=100000)

# Load some sentences
sentences = []
sampleTxt = 'parallelSample.txt'  # 'samples.txt'
with open(sampleTxt) as f:
    for line in f:
        sentences.append(line.strip())
print(len(sentences))

# LookUpTable
embeddings = model.encode(sentences,
                          bsize=128,
                          tokenize=False,
                          verbose=True)
print('nb sentences encoded : {0}'.format(len(embeddings)))


beforeCompre = "Serge Ibaka -- the Oklahoma City Thunder forward who was born in the Congo but played in Spain -- has been granted Spanish citizenship and will play for the country in EuroBasket this summer, the event where spots in the 2012 Olympics will be decided ."
afterCompre = "Serge Ibaka has been granted Spanish citizenship and will play in EuroBasket ."
contradict = "Serge Ibaka has failed to get Spanish citizenship and will not be able to play in EuroBasket ."
nonRelevant = "Serge Ibaka looks pretty ."
encodeBefore = model.encode([beforeCompre])[0]
encodeAfter = model.encode([afterCompre])[0]
encodeContra = model.encode([contradict])[0]
encodeNonRelevant = model.encode([nonRelevant])[0]

similarity = utils.l2dist(encodeBefore, encodeAfter)
print(similarity)

# similarity_compare1 = utils.l2dist(encodeBefore, encodeContra)
# similarity_compare2 = utils.l2dist(encodeAfter, encodeContra)
#
# print(similarity_compare1)
# print(similarity_compare2)

similarity_compare3 = utils.l2dist(encodeBefore, encodeNonRelevant)
similarity_compare4 = utils.l2dist(encodeAfter, encodeNonRelevant)
print(similarity_compare3)
print(similarity_compare4)


if __name__ == "__main__":
    pass
