import logging

from torch import nn, Tensor

import modules
from modules import FormatSentence

logger = logging.getLogger(__name__)


class ModelGraph(nn.Module):
    def __init__(
            self,
            model_max_length: int,
            pos_dim: int,
            char_dim: int,
            word2vec_select: str,
            pretrain_select: str,
            num_heads: int,
            num_iters: int,
            window_size: int,
            use_gate: bool,
            use_hybrid: bool,
            updates: list[str],
            chars_list: list[str],
            pos_list: list[str],
            types2idx: dict,
            idx2types: dict,
            output_attentions: bool = False,
    ):
        super(ModelGraph, self).__init__()
        self.output_attentions = output_attentions
        types_num = len(types2idx)

        encoder_kargs = {
            'types_num': types_num,
            'embedding_kargs': {
                'model_max_length': model_max_length,
                'pos_dim': pos_dim,
                'char_dim': char_dim,
                'word2vec_select': word2vec_select,
                'chars_list': chars_list,
                'pos_list': pos_list,
                'pretrain_select': pretrain_select,
            }
        }
        self.encoder = modules.Encoder(**encoder_kargs)

        graph_kargs = {
            'num_iters': num_iters,
            'layer_kargs': {
                'hidden_size': self.encoder.hidden_size,
                'num_heads': num_heads,
                'window_size': window_size,
                'use_gate': use_gate,
                'use_hybrid': use_hybrid,
                'updates': updates,
            }
        }
        self.graph = modules.Graph(**graph_kargs)

        decoder_kargs = {
            'hidden_size': self.encoder.hidden_size,
            'types_num': types_num,
            'types2idx': types2idx,
            'idx2types': idx2types,
        }
        self.decoder = modules.GraphDecoder(**decoder_kargs)

    def forward(self, batch_sentences: list[FormatSentence]) -> Tensor:
        """

        :param batch_sentences: (batch_size)
        :return: loss
        """
        context, types, mask = self.encoder(batch_sentences)
        context, types = self.graph(context, types, mask)
        return self.decoder(context, types, mask, batch_sentences)

    def get_attentions(self, batch_sentences: list[FormatSentence]):
        context, types, mask = self.encoder(batch_sentences)
        return self.graph.get_attentions(context, types, mask)

    def decode(self, batch_sentences: list[FormatSentence]) -> list[list[dict]]:
        """

        :param batch_sentences: (batch_size, sentence_length)
        :return:  (batch_size, entities_num, 3)
        """
        context, types, mask = self.encoder(batch_sentences)
        context, types = self.graph(context, types, mask)
        return self.decoder.decode(context, types, mask)
