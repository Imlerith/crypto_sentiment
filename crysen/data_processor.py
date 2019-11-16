import warnings
warnings.filterwarnings(action='ignore', category=UserWarning, module='gensim')
import copy
import string
import re
import operator
from collections import Counter
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from keras.utils import to_categorical
from keras.preprocessing.sequence import pad_sequences
from keras.preprocessing.text import Tokenizer
import gensim
from gensim.models.word2vec import Word2Vec


class DataProcessor:
    def __init__(self, messages_df, context_window=None, min_msg_length=2,
                 train_wv=False, vocabulary_size=10000, embedding_dim=None,
                 max_msg_length="auto", min_term_frequency="auto",
                 msg_length_threshold=0.90, term_freq_threshold=0.99,
                 random_state=21):
        assert (context_window is not None and embedding_dim is not None and train_wv) or (not train_wv), \
            "Word2Vec context window and embedding' dimension are required if embeddings are pre-trained"
        self.messages_df = copy.deepcopy(messages_df)
        self.context_window = context_window
        self.min_msg_length = min_msg_length
        self.train_wv = train_wv
        self._vocab_size = vocabulary_size
        if self.train_wv:
            self._return_string = False
        else:
            self._return_string = True
        self.embedding_dim = embedding_dim
        self.__messages_df_with_tokens = None
        self._messages, self._sentiments = self._tokenize_messages()
        self.msg_length_threshold = msg_length_threshold
        self.term_freq_threshold = term_freq_threshold
        self.__all_terms_freqs_sorted_df = None
        self.__msg_lengths_df = None
        if max_msg_length == "auto":
            self.max_msg_length = self._get_optimal_length()
        else:
            self.max_msg_length = max_msg_length
        if max_msg_length == "auto":
            self.min_term_frequency = self._get_term_frequency()
        else:
            self.min_term_frequency = min_term_frequency
        self.random_state = random_state
        self.__word_to_index_map = None
        self.__wv_matrix = None
        if self.train_wv:
            self._labeled_sentence = gensim.models.doc2vec.LabeledSentence
            self._fit_embeddings()

    def _filter_messages(self, text):
        # --- clean the text
        text = re.sub(r"\s{2,}", " ", text)  # remove multiple whitespaces
        text = re.sub(r"\s+t\s+", "'t ", text)  # replace separately standing "t" as 't
        text = re.sub(r"”", "", text)
        text = re.sub(r"“", "", text)
        text = re.sub(r"http stks co \w+\s*", "", text)
        text = re.sub(r"\w+tag_", "", text)
        text = re.sub(r"\w+tag\s*", "", text)
        # --- remove punctuation and whitespaces from both ends
        translator = str.maketrans('', '', string.punctuation)
        text = text.translate(translator).strip()
        # --- convert words to lower case and split them
        text = text.lower().split(" ")
        # --- remove stop words
        my_stops = ['.', ',', "’", ':', ';', '\\', '//', '#',
                    '*', '(', ')', '<', '>', '~', '^', "'",
                    '{', '}', '[', ']', '¿', '|', '"', "&",
                    '-', '/', '_', '`', '', '\t', '\n',
                    "quot", 'http', 'https', 'com',
                    'www', 'an', 'the'] + \
                   list('abcdefghjklmnopqrstuvwxyz')
        tokens = [w for w in text if not w in my_stops]
        tokens_with_unicode_emoticons = list()
        for w in tokens:
            try:
                # --- get Unicode codepoint for an emoticon
                token = 'U+{:X}'.format(ord(w))
                tokens_with_unicode_emoticons.append(token)
            except TypeError:
                tokens_with_unicode_emoticons.append(w)
        if self._return_string:
            return " ".join(tokens_with_unicode_emoticons)
        return tokens_with_unicode_emoticons

    def _tokenize_messages(self):
        # --- tokenize sentences
        messages_df = self.messages_df.loc[:, ['declared_sentiment', 'set', 'created_at', 'body']] \
            .rename(columns={'declared_sentiment': 'sentiment', 'created_at': 'timestamp'}) \
            .reset_index(drop=True)
        messages_df['date'] = messages_df['timestamp'].apply(lambda x: x[:10])
        messages_df['date'] = pd.to_datetime(messages_df['date'])
        # --- tokenize messages
        filtered_messages = messages_df['body'].apply(self._filter_messages).values
        messages_df['message'] = filtered_messages
        self.__messages_df_with_tokens = messages_df
        messages_df = messages_df.dropna()
        if self.train_wv:
            messages_df = messages_df[messages_df['message'].astype(str) != '[]'].reset_index(drop=True)
        else:
            messages_df = messages_df[messages_df['message'].astype(str) != ''].reset_index(drop=True)
        messages = messages_df['message'].values
        sentiments = messages_df['sentiment'].values
        return messages, sentiments

    def _get_optimal_length(self):
        # --- determine optimal length of a message
        counts_msg_lengths = Counter([len(x) for x in self._messages])
        counts_msg_lengths_dict = dict(counts_msg_lengths)
        counts_msg_lengths_dict_sorted = sorted(counts_msg_lengths_dict.items(),
                                                key=operator.itemgetter(0))
        self.__msg_lengths_df = pd.DataFrame.from_records(counts_msg_lengths_dict_sorted).rename(columns={0: 'msg_length',
                                                                                                   1: 'abs_freq'})
        self.__msg_lengths_df = self.__msg_lengths_df.iloc[2:]  # take all messages equal in length or longer than 2
        self.__msg_lengths_df['rel_freq'] = self.__msg_lengths_df['abs_freq'] / self.__msg_lengths_df['abs_freq'].sum()
        self.__msg_lengths_df['cum_perc'] = self.__msg_lengths_df['rel_freq'].cumsum()
        max_msg_length = int(self.__msg_lengths_df[self.__msg_lengths_df['cum_perc'] >
                                                   self.msg_length_threshold].iloc[0].msg_length)
        return max_msg_length

    def _get_term_frequency(self):
        # --- get a table of terms and their frequencies
        all_terms_flattened = [x for sublist in self._messages for x in sublist]
        all_terms_freqs = dict(Counter(all_terms_flattened))
        all_terms_freqs_sorted = sorted(all_terms_freqs.items(),
                                        key=operator.itemgetter(1))
        all_terms_freqs_sorted.reverse()
        self.__all_terms_freqs_sorted_df = pd.DataFrame.from_records(all_terms_freqs_sorted).rename(columns={0: 'term',
                                                                                                      1: 'abs_freq'})
        self.__all_terms_freqs_sorted_df['rel_freq'] = self.__all_terms_freqs_sorted_df['abs_freq'
                                                ] / self.__all_terms_freqs_sorted_df['abs_freq'].sum()
        self.__all_terms_freqs_sorted_df['cum_perc'] = self.__all_terms_freqs_sorted_df['rel_freq'].cumsum()
        # --- get a threshold for how many terms to use according to cumulative frequency
        min_term_freq = self.__all_terms_freqs_sorted_df[self.__all_terms_freqs_sorted_df['cum_perc'] >
                                                         self.term_freq_threshold].iloc[0].abs_freq
        return min_term_freq

    def _fit_embeddings(self):
        # --- train Word2Vec
        x_all = self._label_tokens(self._messages, 'ALL')
        msg_w2v = Word2Vec(size=self.embedding_dim, min_count=self.min_term_frequency,
                           sg=1, window=self.context_window, workers=16)
        msg_w2v.build_vocab([x.words for x in x_all])
        sentences = [x.words for x in x_all]
        msg_w2v.train(sentences=sentences,
                      total_examples=msg_w2v.corpus_count,
                      epochs=25)
        word_vectors = msg_w2v.wv
        # --- get the word-to-index dictionary
        self.__word_to_index_map = {}
        idx = 1
        for word in word_vectors.vocab.keys():
            self.__word_to_index_map[word] = idx
            idx += 1
        # --- get embedding matrix
        vocab_size = len(word_vectors.vocab) + 1
        self.__wv_matrix = np.random.uniform(-0.1, 0.1, size=(vocab_size, self.embedding_dim))
        for word, i in self.__word_to_index_map.items():
            if i >= vocab_size:
                continue
            try:
                embedding_vector = word_vectors[word]
                self.__wv_matrix[i] = embedding_vector
            except Exception:
                pass

    def _label_tokens(self, tweets, label_type):
        labeled = []
        for j, v in enumerate(tweets):
            label = '%s_%s' % (label_type, j)
            labeled.append(self._labeled_sentence(v, [label]))
        return labeled

    def _get_mapped_messages(self):
        # --- filter messages by length and pad them
        if self.train_wv:
            all_sequences_labeled = [([self.word_to_index_map.get(t) for t in msg if t in self.word_to_index_map.keys()], label)
                                     for msg, label in zip(self._messages, self._sentiments)]
            messages_filtered = [x[0] for x in all_sequences_labeled if len(x[0]) > self.min_msg_length]
            sentiments_filtered = [x[1] for x in all_sequences_labeled if len(x[0]) > self.min_msg_length]
            all_seqs_padded = pad_sequences(messages_filtered, maxlen=self.max_msg_length,
                                            padding="post", truncating="post")
        else:
            tokenizer = Tokenizer(num_words=self._vocab_size, filters='\t\n', lower=False)
            tokenizer.fit_on_texts(self._messages)
            self.__word_to_index_map = {k: v for k, v in tokenizer.word_index.items() if v < self._vocab_size}
            all_messages_tokenized = tokenizer.texts_to_sequences(self._messages)
            all_seqs_padded = pad_sequences(all_messages_tokenized, maxlen=self.max_msg_length,
                                            padding="post", truncating="post")
            sentiments_filtered = self._sentiments
        return all_seqs_padded, sentiments_filtered

    def get_train_valid_test_data(self, valid_size=0.1, test_size=0.1):
        sentences, sentiments = self._get_mapped_messages()
        # --- get train and validation data
        sentiments_encoded_01 = [0 if x == -1 else 1 for x in sentiments]
        y_classfn = np.array(sentiments_encoded_01).reshape(-1, 1)
        x_classfn = sentences
        x_train, x_valid, y_train, y_valid = train_test_split(x_classfn, y_classfn, stratify=y_classfn,
                                                              test_size=valid_size, random_state=self.random_state)
        x_train, x_test, y_train, y_test = train_test_split(x_train, y_train, stratify=y_train,
                                                            test_size=test_size/(1 - valid_size),
                                                            random_state=self.random_state)
        y_train_c = to_categorical(y_train, num_classes=2)
        y_valid_c = to_categorical(y_valid, num_classes=2)
        y_test_c = to_categorical(y_test, num_classes=2)
        share_of_bearish = sum([1 for x in y_train.ravel() if x == 0]) / len(y_train)
        print(f"Share of bearish in the train set before upsampling: {share_of_bearish}")
        return x_train, x_valid, x_test, y_train_c, y_valid_c, y_test_c

    def get_upsampled_data(self, x_train, y_train, up_ratio=0.70):
        # --- upsample bearish messages
        y_train_one = np.argmax(y_train, 1)
        bearish_idx = np.where(y_train_one == 0)[0]
        bullish_idx = np.where(y_train_one == 1)[0]
        n_bullish = len(bullish_idx)
        np.random.seed(self.random_state)
        bearish_idx_upsampled = np.random.choice(bearish_idx, size=int(n_bullish * up_ratio), replace=True)
        y_train_up = np.concatenate((y_train_one[bearish_idx_upsampled], y_train_one[bullish_idx]))
        x_train_up = np.concatenate((x_train[bearish_idx_upsampled], x_train[bullish_idx]))
        share_of_bearish = sum([1 for x in y_train_up.ravel() if x == 0]) / len(y_train_up)
        print(f"Share of bearish in the train set after upsampling: {share_of_bearish}")
        y_train_up_c = to_categorical(y_train_up, num_classes=2)
        return x_train_up, y_train_up_c

    @property
    def word_to_index_map(self):
        return self.__word_to_index_map

    @property
    def vocab_size(self):
        return len(self.__word_to_index_map) + 1

    @property
    def wv_matrix(self):
        return self.__wv_matrix

    @property
    def all_terms_freqs_sorted_df(self):
        if self.__all_terms_freqs_sorted_df is not None:
            return self.__all_terms_freqs_sorted_df
        else:
            _ = self._get_term_frequency()
            return self.__all_terms_freqs_sorted_df

    @property
    def msg_lengths_df(self):
        if self.__msg_lengths_df is not None:
            return self.__msg_lengths_df
        else:
            _ = self._get_optimal_length()
            return self.__msg_lengths_df

    @property
    def messages_df_with_tokens(self):
        return self.__messages_df_with_tokens
