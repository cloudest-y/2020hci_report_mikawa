import os
from .general_utils import get_logger
from .data_utils import get_trimmed_glove_vectors, load_vocab, \
        get_processing_word


class Config():
    def __init__(self, load=True):
        """Initialize hyperparameters and load vocabs

        Args:
            load_embeddings: (bool) if True, load embeddings into
                np array, else None

        """
        # directory for training outputs
        if not os.path.exists(self.dir_output):
            os.makedirs(self.dir_output)

        # create instance of logger
        self.logger = get_logger(self.path_log)

        # load if requested (default)
        if load:
            self.load()

    def load(self):
        """Loads vocabulary, processing functions and embeddings

        Supposes that build_data.py has been run successfully and that
        the corresponding files have been created (vocab and trimmed GloVe
        vectors)

        """
        # 1. vocabulary
        self.vocab_words = load_vocab(self.filename_words)
        self.vocab_tags  = load_vocab(self.filename_tags)
        self.vocab_chars = load_vocab(self.filename_chars)

        self.nwords     = len(self.vocab_words)
        self.nchars     = len(self.vocab_chars)
        self.ntags      = len(self.vocab_tags)

        # 2. get processing functions that map str -> id
        self.processing_word = get_processing_word(self.vocab_words,
                self.vocab_chars, lowercase=True, chars=self.use_chars)
        self.processing_tag  = get_processing_word(self.vocab_tags,
                lowercase=False, allow_unk=False)

        # 3. get pre-trained embeddings
        self.embeddings = (get_trimmed_glove_vectors(self.filename_trimmed)
                if self.use_pretrained else None)


    # general config
    dir_output = "results/test/"
    dir_model  = dir_output
    path_log   = dir_output + "log.txt"

    # embeddings
    dim_word = 200
    dim_char = 200

    # w2v files
    filename_glove = "data/jawiki.word_vectors.200d.txt"
    filename_trimmed = "data/jawiki.word_vectors.{}d.trimmed.npz".format(dim_word)
    use_pretrained = True

    #dataset
    filename_path = "data/"
    filename_dev = filename_path + "valid.txt"
    filename_test = filename_path + "test.txt"
    filename_train = filename_path + "train.txt"

    max_iter = None # if not None, max number of examples in Dataset

    # vocab (created from dataset with build_data.py)
    filename_words = filename_path + "words.txt"
    filename_tags = filename_path + "tags.txt"
    filename_chars = filename_path + "chars.txt"

    # training
    train_embeddings = False
    nepochs          = 10
    dropout          = 0.5
    batch_size       = 16
    lr_method        = "adam"
    lr               = 0.001
    lr_decay         = 0.9
    epoch_drop       = 1 # Step Decay: per # epochs to apply lr_decay
    clip             = -1 # if negative, no clipping
    nepoch_no_imprv  = 3

    # model hyperparameters
    hidden_size_char = 100 # lstm on chars
    hidden_size_lstm = 300 # lstm on word embeddings

    ner_model_path = "saves/jawiki{}word_elmo_1024d".format(nepochs)
#     ner_model_path = "saves/jawiki{}word_lstm_200d".format(nepochs)

    # ner_ft_path = "saves/60wiki30net_{}ft_word_elmo_1024d".format(nepochs)


    # elmo config
#     use_elmo = False
    dim_elmo = 1024
    use_elmo = True

    # NOTE: if both chars and crf, only 1.6x slower on GPU
    use_crf = True # if crf, training is 1.7x slower on CPU
    use_chars = False if use_elmo else True#  if char embedding, training is 3.5x slower on CPU
    # use_chars =  True #  if char embedding, training is 3.5x slower on CPU
