import transformers

class BinaryTokenizer(transformers.PreTrainedTokenizer):

    def __call__(self, text, text_pair = None, *params, **kwparams):
        is_batched = isinstance(text, (list, tuple))
        if is_batched:
            batch_text_or_text_pairs = list(zip(text, text_pair)) if text_pair is not None else text
            return self.batch_encode_plus(
                batch_text_or_text_pairs=batch_text_or_text_pairs,
                *params, **kwparams
            )
        else:
            return self.encode_plus(
                text=text,
                text_pair=text_pair,
                *params,
                **kwparams
             )

    def _encode_plus(self, text, text_pair = None, *params, **kwparams):
        batched_input = [(text, text_pair)] if text_pair else [text]
        return self._batch_encode_plus(batched_input, *params, **kwparams)
        #first_ids = self.tokenize(text)
        #second_ids = text_pair and self.tokenize(text_pair)
        #return self.prepare_for_model(first_ids, pair_ids=second_ids, *params, **kwparams)

    def _batch_encode_plus(self, batch_text_or_text_pairs, *params, is_split_into_words = False, return_offsets_mapping = False, **kwparams):
        input_ids = []
        for ids_or_pair_ids in batch_text_or_text_pairs:
            if not isinstance(ids_or_pair_ids, (list, tuple)):
                ids, pair_ids = ids_or_pair_ids, None
            else:
                ids, pir_ids = ids_or_pair_ids

            first_ids = self.tokenize(ids)
            second_ids = pair_ids and self.tokenize(pair_ids)
            input_ids.append((first_ids, second_ids))
        batch_outputs = self._batch_prepare_for_model(input_ids, *params, **kwparams)
        return transformers.tokenization_utils_base.BatchEncoding(batch_outputs)
        
    def tokenize(self, text, **kwargs):
        if type(text) is str:
            text = text.encode()
        return list(text)

    def convert_ids_to_tokens(self, index, skip_special_tokens=True):
        if type(index) is int:
            index = (index,)
        return [bytes((val,)) if val < 256 else b'' for val in index]
    def convert_tokens_to_string(self, tokens):
        return b''.join(tokens)

    def _decode(self, token_ids, skip_special_tokens = False, clean_up_tokenization_spaces = False, spaces_between_special_tokens = False, **kwparams):
        return self.convert_tokens_to_string(self.convert_ids_to_tokens(token_ids))
        #filtered_tokens = self.convert_ids_to_tokens(token_ids, skip_special_tokens=skip_special_tokens)
        #sub_texts
