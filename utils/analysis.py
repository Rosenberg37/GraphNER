import logging

import torch
from bertviz import head_view
from torch import nn
from tqdm import tqdm

import config
import modules
import utils
from modules import FormatCorpus, FormatSentence

logger = logging.getLogger(__name__)


class Analysist:
    def __init__(self, corpus: FormatCorpus, model: nn.Module):
        self.corpus = corpus
        self.model = model
        self.statistics = None
        self.clean()

        self.length_cut = 4

    def clean(self):
        self.statistics = {
            'all': {'TP': 0, 'FP': 0, 'FN': 0},
            'length': {},
            'nest': {
                'NST': {'TP': 0, 'FP': 0, 'FN': 0},
                'NDT': {'TP': 0, 'FP': 0, 'FN': 0},
                'flat': {'TP': 0, 'FP': 0, 'FN': 0},
            }

        }

    def update(self, sentence: FormatSentence, predict_entities: list[dict], real_entities: list[dict]) -> dict:
        def giou(entity_1: dict, entity_2: dict):
            max_s, max_e = max(entity_1['start'], entity_2['start']), max(entity_1['end'], entity_2['end'])
            min_s, min_e = min(entity_1['start'], entity_2['start']), min(entity_1['end'], entity_2['end'])
            return max((min_e - max_s) / (max_e - min_s), 0)

        for entity in predict_entities:
            if entity in real_entities:
                self.statistics['all']['TP'] += 1

                entity_length = entity['end'] - entity['start']
                if entity_length > self.length_cut:
                    entity_length = self.length_cut + 1

                if entity_length not in self.statistics['length'].keys():
                    self.statistics['length'][entity_length] = {'TP': 0, 'FP': 0, 'FN': 0}
                self.statistics['length'][entity_length]['TP'] += 1

                if entity in sentence.get_nested_entities('NST'):
                    self.statistics['nest']['NST']['TP'] += 1
                if entity in sentence.get_nested_entities('NDT'):
                    self.statistics['nest']['NDT']['TP'] += 1
                if entity not in sentence.nested_entities:
                    self.statistics['nest']['flat']['TP'] += 1
            else:
                self.statistics['all']['FP'] += 1

                if len(real_entities) > 0:
                    pair_entity = max(real_entities, key=lambda a: giou(a, entity))
                    entity_length = pair_entity['end'] - pair_entity['start']
                    if entity_length > self.length_cut:
                        entity_length = self.length_cut + 1

                    if entity_length not in self.statistics['length'].keys():
                        self.statistics['length'][entity_length] = {'TP': 0, 'FP': 0, 'FN': 0}
                    self.statistics['length'][entity_length]['FP'] += 1

                    if pair_entity in sentence.get_nested_entities('NST'):
                        self.statistics['nest']['NST']['FP'] += 1
                    if pair_entity in sentence.get_nested_entities('NDT'):
                        self.statistics['nest']['NDT']['FP'] += 1
                    if pair_entity not in sentence.nested_entities:
                        self.statistics['nest']['flat']['FP'] += 1

        for entity in real_entities:
            entity_length = entity['end'] - entity['start']
            if entity_length > self.length_cut:
                entity_length = self.length_cut + 1

            if entity_length not in self.statistics['length'].keys():
                self.statistics['length'][entity_length] = {'TP': 0, 'FP': 0, 'FN': 0}

            if entity not in predict_entities:
                self.statistics['all']['FN'] += 1
                self.statistics['length'][entity_length]['FN'] += 1

                if entity in sentence.get_nested_entities('NST'):
                    self.statistics['nest']['NST']['FN'] += 1
                if entity in sentence.get_nested_entities('NDT'):
                    self.statistics['nest']['NDT']['FN'] += 1
                if entity not in sentence.nested_entities:
                    self.statistics['nest']['flat']['FN'] += 1
        return self.statistics

    def analysis(self, clean: bool = True) -> dict[str, dict]:
        def transform(values: dict):
            tp, fp, fn = values.pop('TP'), values.pop('FP'), values.pop('FN')
            p = tp / (tp + fp) if tp != 0 else 0
            r = tp / (tp + fn) if tp != 0 else 0
            f1 = 2 * p * r / (p + r) if p * r != 0 else 0
            values['precision'], values['recall'], values['f1'] = p, r, f1

        transform(self.statistics['all'])
        for values in self.statistics['length'].values():
            transform(values)
        for values in self.statistics['nest'].values():
            transform(values)

        statistic = self.statistics.copy()
        if clean:
            self.clean()
        return statistic

    def evaluate(self, name: str) -> dict[str, dict]:
        """

        :param name: select which dataset to evaluate on
        :return: statistic
        """
        self.model.eval()
        with torch.no_grad():
            t = tqdm(self.corpus.__dict__[f'_{name}'], desc="Evaluation")
            for sentence in t:
                real_entities = sentence.entities
                predict_entities = self.model.decode([sentence])[0]
                self.statistics = self.update(sentence, predict_entities, real_entities)
                t.set_postfix({'statistics': self.statistics['all']})
        self.model.train()
        return self.analysis(clean=True)

    def scheme_error(self, names: list[str], print_instance: bool = False):
        tagger = modules.Tagger(self.corpus.metadata['types2idx'], self.corpus.metadata['idx2types'])
        types_num = len(self.corpus.metadata['types2idx'])
        count_sentence, fn, fp = 0, 0, 0

        for name in names:
            dataset = self.corpus.__dict__[f'_{name}']
            for i, sentence in tqdm(enumerate(dataset), desc=f"Error on {name}"):
                batch_tags = tagger([sentence], [1, types_num, len(sentence)])
                entities = tagger.detagging(batch_tags)[0]
                false_positive = list(filter(lambda a: a not in entities, sentence.entities))
                false_negative = list(filter(lambda a: a not in sentence.entities, entities))

                fn += len(false_negative)
                fp += len(false_positive)
                if len(false_positive) != 0 or len(false_negative) != 0:
                    count_sentence += 1
                    if print_instance:
                        print(
                            f"{i} {sentence}:\n"
                            f"Nested entities:{sentence.nested_entities}\n"
                            f"Real entities:{sentence.entities}\n"
                            f"Predicted entities:{entities}\n"
                            f"FP:{false_positive}\n"
                            f"FN:{false_negative}\n"
                        )

            print(
                f"{name}:\n"
                f"Complex sentences:{count_sentence}\n"
                f"FP:{fp}\n",
                f"FN:{fn}\n",
            )

    def case_study(self, case_condition: callable):
        self.model.eval()
        for i, sentence in enumerate(filter(case_condition, self.corpus.test)):
            entities = self.model.decode([sentence])[0]

            equal = True
            for entity in entities:
                if entity not in sentence.entities:
                    equal = False
            for entity in sentence.entities:
                if entity not in entities:
                    equal = False
            if not equal:
                continue
            print(f"{i} {sentence}\n"
                  f"Nested Pairs:{sentence.nested_pairs}\n"
                  f"Real Entities:{sentence.entities}\n"
                  f"Predict Entities:{entities}\n")

    def visualize(self, sentence: FormatSentence, types: list[str], select: str, color_depth: float = 1):
        context_attentions, types_attentions = self.model.get_attentions([sentence])
        context_attentions.append(sum(context_attentions) / len(context_attentions))
        types_attentions.append(sum(types_attentions) / len(types_attentions))
        tokens = types + sentence.sentence_tokens

        def color_func(color_factor):
            return color_factor * color_depth

        if 'context' == select:
            head_view(list(map(color_func, context_attentions)), tokens)
        if 'types' == select:
            head_view(list(map(color_func, types_attentions)), tokens)


if __name__ == '__main__':
    corpus = modules.FormatCorpus(**vars(config.DATA_OPTIONS))
    model = utils.init_model(corpus)
    analysist = utils.Analysist(corpus, model)
    analysist.scheme_error(['train', 'test', 'dev'])
