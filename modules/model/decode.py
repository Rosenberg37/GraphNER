import math
from enum import IntEnum
from typing import Optional, Union

import torch
import torch.nn as nn
from torch import Tensor
from torch.nn.utils.rnn import pack_padded_sequence, PackedSequence, pad_packed_sequence, pad_sequence

import modules
from modules import FormatSentence


class Tags(IntEnum):
    begin = 0
    inside = 1
    out = 2
    end = 3
    single = 4


class GraphDecoder(nn.Module):
    def __init__(self, hidden_size: int, types_num: int, types2idx: dict, idx2types: dict):
        super(GraphDecoder, self).__init__()
        self.tagger = Tagger(types2idx, idx2types)

        self.rnns = nn.ModuleList([nn.GRU(hidden_size, hidden_size, batch_first=True, bidirectional=True) for _ in range(types_num)])
        self.reduces = nn.Parameter(torch.empty(types_num, hidden_size * 2, hidden_size))
        nn.init.kaiming_uniform_(self.reduces, a=math.sqrt(5))
        self.scores = nn.Parameter(torch.empty(types_num, hidden_size, len(Tags)))
        nn.init.kaiming_uniform_(self.scores, a=math.sqrt(5))

        self.crf = modules.NestCRF()

    def emission(self, types: Tensor, context: Tensor, lengths: Tensor):
        """

        :param lengths: [batch_size]
        :param types: [batch_size, types_num, hidden_size]
        :param context: [batch_size, sentence_length, hidden_size]
        :return: [batch_size, types_num, sentence_length, num_tags]
        """
        pack_context = pack_padded_sequence(context, lengths, enforce_sorted=False, batch_first=True)
        rnn_hiddens = torch.repeat_interleave(types.unsqueeze(0), 2, dim=0)  # [2, batch_size, types_num, hidden_size]

        batch_hiddens = list()
        for rnn, hidden in zip(self.rnns, rnn_hiddens.permute(2, 0, 1, 3)):
            packed_hiddens = rnn(pack_context, hidden)[0]
            batch_hiddens.append(pad_packed_sequence(packed_hiddens, batch_first=True)[0])
        batch_hiddens = torch.stack(batch_hiddens, dim=1)
        batch_hiddens = torch.einsum('btlh,thd->btld', batch_hiddens, self.reduces)
        batch_hiddens = torch.maximum(batch_hiddens + context.unsqueeze(1), types.unsqueeze(2))
        return torch.einsum('btld,tdg->btlg', batch_hiddens, self.scores)

    def forward(self, types: Tensor, context: Tensor, context_mask: Tensor, batch_sentences: list[FormatSentence]):
        """

        :param batch_sentences: (batch_size, sentence_length)
        :param types: [batch_size, types_num, hidden_size]
        :param context: [batch_size, sentence_length, hidden_size]
        :param context_mask: [batch_size, sentence_length]
        :return: loss
        """
        batch_scores = self.emission(types, context, torch.sum(context_mask, dim=-1).cpu())
        batch_tags = self.tagger(batch_sentences, batch_scores.shape[:3], batch_scores.device)
        mask = context_mask.unsqueeze(1).expand_as(batch_tags)
        return self.crf(*map(lambda a: a.flatten(0, 1), [batch_scores, batch_tags, mask]))

    def decode(self, types: Tensor, context: Tensor, context_mask: Tensor):
        """

        :param types: [batch_size, types_num, hidden_size]
        :param context: [batch_size, types_num, sentence_length, hidden_size]
        :param context_mask: [batch_size, sentence_length]
        :return: (batch_size, entities_num, 3)
        """
        batch_scores = self.emission(types, context, torch.sum(context_mask, dim=-1).cpu())
        mask = context_mask.unsqueeze(1).expand(batch_scores.shape[:-1])
        batch_tags = self.crf.decode(*map(lambda a: a.flatten(0, 1), [batch_scores, mask]))
        batch_tags = [torch.as_tensor(tags) for tags in batch_tags]
        batch_tags = pad_sequence(batch_tags, batch_first=True, padding_value=float(Tags.out))
        batch_tags = batch_tags.view(batch_scores.shape[:-1]).long()
        return self.tagger.detagging(batch_tags)


class ScoreBlock(nn.Module):
    def __init__(self, hidden_size: int, num_tags: int):
        super().__init__()
        self.rnn = nn.GRU(hidden_size, hidden_size, batch_first=True, bidirectional=True)
        self.reduce = nn.Linear(hidden_size * 2, hidden_size)
        self.score = nn.Linear(hidden_size, num_tags)

    def forward(self, type_embed: Tensor, pack_context: PackedSequence, context: Tensor):
        """

        :param type_embed: [batch_size, hidden_size]
        :param context: [batch_size, sentence_length, hidden_size]
        :param pack_context: [batch_size, sentence_length, hidden_size] as packed sequence
        :return: [batch_size, types_num, sentence_length, num_tags]
        """
        batch_hiddens = self.rnn(pack_context, torch.repeat_interleave(type_embed.unsqueeze(0), 2, dim=0))[0]
        batch_hiddens = pad_packed_sequence(batch_hiddens, batch_first=True)[0]
        batch_hiddens = type_embed.unsqueeze(1) + context + self.reduce(batch_hiddens)
        return self.score(batch_hiddens)


class Tagger:
    def __init__(self, types2idx: dict, idx2types: dict):
        super(Tagger, self).__init__()
        self.types2idx, self.idx2types = types2idx, idx2types

    def __call__(self, batch_sentences: list[FormatSentence], shape: Union[torch.Size, list[int]], device: str = 'cpu') -> Tensor:
        """

        :param device: device sentence created on.
        :param shape: batch_size, types_num, sentence_length
        :param batch_sentences: (batch_size, entities_num)
        :return:  (batch_size, entities_num, 3)
        """
        batch_tags = torch.full(shape, Tags.out, device=device, dtype=torch.long)
        for sentence, types_tags in zip(batch_sentences, batch_tags):
            entities = list(filter(lambda a: a['end'] <= shape[-1], sentence.entities))
            triples = list(map(lambda a: (a['start'], a['end'], self.types2idx[a['type']]), entities))
            for s, e, t in triples:
                types_tags[t, s:e] = Tags.inside
            for s, e, t in filter(lambda a: a[0] != a[1] - 1, triples):
                types_tags[t, s], types_tags[t, e - 1] = Tags.begin, Tags.end
            for s, e, t in filter(lambda a: a[0] == a[1] - 1, triples):
                types_tags[t, s] = Tags.single

        return batch_tags

    def detagging(self, batch_tags: Tensor) -> list[list[dict]]:
        """

        :param batch_tags: [batch_size, types_num, sentence_length]
        :return: (batch_size, entities_num, 1 + 1 + 1(start, end, type))
        """

        batch_entities = list()
        for types_tags in batch_tags:
            entities = list()
            for t, tags in enumerate(types_tags):
                starts = [i for i, tag in enumerate(tags) if tag in [Tags.begin, Tags.single]]
                ends = [i for i, tag in enumerate(tags) if tag in [Tags.end, Tags.single]]
                tags = torch.cat([tags, torch.as_tensor([Tags.out], device=tags.device)])

                while len(starts) != 0 and len(ends) != 0:
                    for s in starts:
                        for e in ends:
                            if s == e:
                                entities.append({'start': s, 'end': e + 1, 'type': self.idx2types[t]})
                                if tags[s + 1] == Tags.out and tags[s - 1] in [Tags.begin, Tags.inside]:
                                    tags[s] = Tags.end
                                    starts.remove(s)
                                elif tags[s + 1] in [Tags.end, Tags.inside] and tags[s - 1] == Tags.out:
                                    tags[s] = Tags.begin
                                    ends.remove(e)
                                else:
                                    starts.remove(s), ends.remove(e)
                                    if tags[s + 1] in [Tags.out, Tags.single] and tags[s - 1] in [Tags.out, Tags.single]:
                                        tags[s] = Tags.out
                                    else:
                                        tags[s] = Tags.inside
                                break
                            elif tags[s] == Tags.begin and tags[e] == Tags.end and \
                                    sum(map(lambda a: a == Tags.inside, tags[s + 1:e])) == e - s - 1:
                                entities.append({'start': s, 'end': e + 1, 'type': self.idx2types[t]})
                                if tags[s - 1] in [Tags.end, Tags.out, Tags.single]:
                                    ends.remove(e)
                                    tags[e] = Tags.inside if tags[e + 1] in [Tags.inside, Tags.end] else Tags.out
                                if tags[e + 1] in [Tags.begin, Tags.out, Tags.single]:
                                    starts.remove(s)
                                    tags[s] = Tags.inside if tags[s - 1] in [Tags.inside, Tags.begin] else Tags.out
                                if tags[s - 1] in [Tags.inside, Tags.begin] and tags[e + 1] in [Tags.inside, Tags.end]:
                                    starts.remove(s), ends.remove(e)
                                    tags[s] = tags[e] = Tags.inside
                                break
                        else:
                            continue
                        break
            batch_entities.append(entities)

            return batch_entities


class NestCRF(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.num_tags = len(Tags)
        self.register_buffer('start_transitions', torch.zeros(self.num_tags))
        self.register_buffer('end_transitions', torch.zeros(self.num_tags))
        self.register_buffer('transitions', torch.zeros(self.num_tags, self.num_tags))

        for tag in ['inside', 'end']:
            self.start_transitions[Tags[tag]] = -1e12

        for tag in ['begin', 'inside']:
            self.end_transitions[Tags[tag]] = -1e12

        pairs = {
            'begin': ['out'],
            'inside': ['out'],
            'out': ['inside', 'end']
        }
        for pre in pairs.keys():
            for post in pairs[pre]:
                self.transitions[Tags[pre], Tags[post]] = -1e12

    def forward(self, emissions: Tensor, tags: Tensor, mask: Optional[Tensor] = None) -> torch.Tensor:
        """
        Compute the conditional log likelihood of a sequence of tags given emission scores.
        :param emissions: (`~torch.Tensor`): Emission score tensor of size
                ``(seq_length, batch_size, num_tags)`` if ``batch_first`` is ``False``,
                ``(batch_size, seq_length, num_tags)`` otherwise.
        :param tags: (`~torch.LongTensor`): Sequence of tags tensor of size
                ``(seq_length, batch_size)`` if ``batch_first`` is ``False``,
                ``(batch_size, seq_length)`` otherwise.
        :param mask: (`~torch.ByteTensor`): Mask tensor of size ``(seq_length, batch_size)``
                if ``batch_first`` is ``False``, ``(batch_size, seq_length)`` otherwise.
        :return: `~torch.Tensor`: The negative log likelihood. This will have size ``(batch_size,)`` if
            reduction is ``none``, ``()`` otherwise.
        """
        if mask is None:  # 0 means padding, 1 means not padding.
            mask = torch.ones_like(tags)

        # transform batch_size dimension to the dimension 1.
        emissions = emissions.transpose(0, 1)
        tags = tags.transpose(0, 1)
        mask = mask.transpose(0, 1)

        # shape: (batch_size,)
        numerator = self._compute_score(emissions, tags, mask)
        # shape: (batch_size,)
        denominator = self._compute_normalizer(emissions, mask)
        # shape: (batch_size,)
        llh = denominator - numerator

        return torch.mean(llh / mask.sum(0))

    def decode(self, emissions: torch.Tensor, mask: Optional[torch.ByteTensor] = None) -> list[list[int]]:
        """
        Find the most likely tag sequence using Viterbi algorithm.
        :param emissions: (`~torch.Tensor`): Emission score tensor of size
                ``(seq_length, batch_size, num_tags)`` if ``batch_first`` is ``False``,
                ``(batch_size, seq_length, num_tags)`` otherwise.
        :param mask: (`~torch.ByteTensor`): Mask tensor of size ``(seq_length, batch_size)``
                if ``batch_first`` is ``False``, ``(batch_size, seq_length)`` otherwise.
        :return: List of list containing the best tag sequence for each batch.
        """
        if mask is None:
            mask = emissions.new_ones(emissions.shape[:2], dtype=torch.long)

        emissions = emissions.transpose(0, 1)
        mask = mask.transpose(0, 1)

        return self._viterbi_decode(emissions, mask)

    def _compute_score(self, emissions: torch.Tensor, tags: torch.LongTensor, mask: torch.ByteTensor) -> torch.Tensor:
        """

        :param emissions: (seq_length, batch_size, num_tags)
        :param tags: (seq_length, batch_size)
        :param mask: (seq_length, batch_size)
        :return:
        """

        seq_length, batch_size = tags.shape

        score = self.start_transitions[tags[0]]
        score += emissions[0, torch.arange(batch_size), tags[0]]

        for i in range(1, seq_length):
            score += self.transitions[tags[i - 1], tags[i]] * mask[i]
            score += emissions[i, torch.arange(batch_size), tags[i]] * mask[i]

        seq_ends = mask.sum(dim=0) - 1
        last_tags = tags[seq_ends, torch.arange(batch_size)]
        score += self.end_transitions[last_tags]

        return score

    def _compute_normalizer(self, emissions: torch.Tensor, mask: torch.ByteTensor) -> torch.Tensor:
        """

        :param emissions: (seq_length, batch_size, num_tags)
        :param mask: (seq_length, batch_size)
        :return:
        """

        seq_length = emissions.shape[0]

        score = self.start_transitions + emissions[0]

        for i in range(1, seq_length):
            broadcast_score = score.unsqueeze(2)
            broadcast_emissions = emissions[i].unsqueeze(1)
            next_score = broadcast_score + self.transitions + broadcast_emissions
            next_score = torch.logsumexp(next_score, dim=1)
            score = torch.where(mask[i].unsqueeze(1), next_score, score)

        score += self.end_transitions

        return torch.logsumexp(score, dim=1)

    def _viterbi_decode(self, emissions: torch.FloatTensor, mask: torch.ByteTensor) -> list[list[int]]:
        """

        :param emissions: (seq_length, batch_size, num_tags)
        :param mask: (seq_length, batch_size)
        :return:
        """

        seq_length, batch_size = mask.shape

        score = self.start_transitions + emissions[0]
        history = []

        for i in range(1, seq_length):
            broadcast_score = score.unsqueeze(2)
            broadcast_emission = emissions[i].unsqueeze(1)
            next_score = broadcast_score + self.transitions + broadcast_emission
            next_score, indices = next_score.max(dim=1)
            score = torch.where(mask[i].unsqueeze(1), next_score, score)
            history.append(indices)

        score += self.end_transitions
        seq_ends = mask.long().sum(dim=0) - 1
        best_tags_list = []

        for idx in range(batch_size):
            _, best_last_tag = score[idx].max(dim=0)
            best_tags = [best_last_tag.item()]

            for hist in reversed(history[:seq_ends[idx]]):
                best_last_tag = hist[idx][best_tags[-1]]
                best_tags.append(best_last_tag.item())

            best_tags.reverse()
            best_tags_list.append(best_tags)

        return best_tags_list
