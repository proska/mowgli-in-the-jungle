import logging
import re
from collections import deque
from typing import List, Union, Iterable, Generator

from mowgli.utils.graphs.Lexicalization.Lexicalizations import PathLexicalizer, EntityLexicError
from utils.graphs.KnowledgeGraphBase import NODE_T, EDGE_T, GRAPH_T

logger = logging.getLogger(__name__)


class RuleBasedLexic(PathLexicalizer):
    def convert(self, path: List[str]) -> str:
        return self._lexicalize_path(path)
    
    def _lexicalize_path(self, random_path: List[str]) -> str:
        cleaned = map(self._clean_relation_entity, random_path)
        edges = self._separate_edges(cleaned)
        sents = map(lambda t: self._triple2str(*t), edges)
        lexic = f", and ".join(sents) + '.\n'
        return lexic

    @staticmethod
    def _triple2str(s: str, p: str, e: str) -> str:
        return f'{s} {RuleBasedLexic._pred2str(p)} {e}'

    @staticmethod
    def _pred2str(pred: str) -> str:
        verb = COMMONSENSE_MAPPING.get(pred, None)
        if verb is None:
            logger.error(f'New predicate: {pred}')
            return pred
        else:
            return verb

    @staticmethod
    def _separate_edges(path: Iterable[str]) -> Generator[List[str], None, None]:
        d = deque(maxlen=3)
        iterable = iter(path)
        for i, it in enumerate(iterable):
            d.append(it)
            if len(d) == 3:
                yield list(d)
                d.popleft()
                d.popleft()

    @staticmethod
    def _clean_relation_entity(s: str) -> str:
        if '/' not in s:
            return s
        for p in [r'\/c\/en\/([^\s\/]*)', r'\/r\/([^\s\/]*)', r'[^:]:([^\s\/]*)']:
            m = re.findall(p, s)
            if len(m) > 0:
                # assert len(m) == 1, f'multiple match (p={p}) in {s} :{m}'
                return m[0].replace('_', ' ')
        raise EntityLexicError(f'{s}')
