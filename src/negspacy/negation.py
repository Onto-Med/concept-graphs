import pathlib
from itertools import islice
from typing import Union, Optional

from spacy.language import Language
from spacy.tokens import Doc, Span, SpanGroup
from spacy.matcher import PhraseMatcher
import logging

from .utils import FeaturesOfInterest, LeftsDependencyLabels, RightsDependencyLabels
from .termsets import termset

default_ts = termset("en_clinical").get_patterns()


@Language.factory(
    "negex",
    default_config={
        "neg_termset": default_ts,
        "feat_types": list(),
        "extension_name": "negex",
        "chunk_prefix": list(),
        "neg_termset_file": None,
        "feat_of_interest": FeaturesOfInterest.NAMED_ENTITIES,
        "scope": None,
        "language": None
    },
)
class Negex:
    """
        A spaCy pipeline component which identifies negated tokens in text.

        Based on: NegEx - A Simple Algorithm for Identifying Negated Findings and Diseasesin Discharge Summaries
    Chapman, Bridewell, Hanbury, Cooper, Buchanan

    Parameters
    ----------
    nlp: object
        spaCy language object
    name: str
        ???
    neg_termset: dict
        a dictionary that defines the negation triggers
    feat_types: list
        list of entity/noun chunk types to negate
    extension_name: str
        defaults to "negex"; whether entity is negated is then available as ent._.negex
    chunk_prefix: list
        list of tokens that can occur in a noun chunk and negate the noun (e.g. ['no'] -> 'no headache')
    neg_termset_file: str | pathlib.Path
        a file that is used in the original negex implementation to define negation triggers;
         if provided it will take precedence over 'neg_termset' and the content will be mapped to a negspacy termset
    feat_of_interest: list[str]
        list of linguistic features under scrutiny; default is only NamedEntities (['ents']) but there might be cases,
         where one wants to check whether NounChunks (['noun_chunks']) are negated or both (['ents', 'noun_chunks'])
         negex has a pseudo enum class 'FeaturesOfInterest' to account for all three cases
    scope: Union[str, int, bool]
        whether dependency parse information shall be used, to limit the scope of a negation;
         if an integer (or digit string for that matter) is supplied it provides the limit
         to the number of dependent children under scrutiny. If set to true, scope is 1
    language: str
    """
    @classmethod
    def set_extension(cls, ext_name: str):
        if not Span.has_extension(ext_name):
            Span.set_extension(ext_name, default=False, force=True)

    def __init__(
            self,
            nlp: Language,
            name: str,
            neg_termset: dict,
            feat_types: list,
            extension_name: str,
            chunk_prefix: list,
            neg_termset_file: Union[pathlib.Path, str, None],
            feat_of_interest: list[str],
            scope: Union[str, int, bool, None],
            language: Optional[str]
    ):
        # if not termset_lang in LANGUAGES:
        #     raise KeyError(
        #         f"{termset_lang} not found in languages termset. "
        #         "Ensure this is a supported termset or specify "
        #         "your own termsets when initializing Negex."
        #     )
        # termsets = LANGUAGES[termset_lang]
        Negex.set_extension(extension_name)
        ts = neg_termset
        if neg_termset_file is not None:
            rules = None
            if isinstance(neg_termset_file, str):
                rules = pathlib.Path(neg_termset_file).read_text().splitlines()
            elif isinstance(neg_termset_file, pathlib.Path):
                rules = neg_termset_file.read_text().splitlines()
            else:
                logging.info("'neg_termset_file' could not be read. Reverting to default 'neg_termset'.")

            if rules is not None:
                _map = {"[CONJ]": "termination", "[PSEU]": "pseudo_negations",
                        "[POST]": "following_negations", "[PREN]": "preceding_negations",
                        "[PREP]": "preceding_speculation", "[POSP]": "following_speculation"}
                ts = {
                    "pseudo_negations": [],
                    "preceding_negations": [],
                    "following_negations": [],
                    "termination": [],
                    "preceding_speculation": [],
                    "following_speculation": [],
                    "none": []
                }
                for rule in rules:
                    _str, _tag = rule.split('\t\t')
                    ts[_map.get(_tag, "none")].append(_str)
        expected_keys = [
            "pseudo_negations",
            "preceding_negations",
            "following_negations",
            "termination",
        ]
        if len(set(ts.keys()).intersection(expected_keys)) != len(expected_keys):
            raise KeyError(
                f"Missing keys in 'neg_termset', expected: {expected_keys}, instead got: {list(ts.keys())}"
            )
        else:
            if len(ts.keys()) > len(expected_keys):
                logging.warning(
                    f"There are trigger types in the termset that are not expected by negspacy and won't be processed:"
                    f" {set(ts.keys()).difference(expected_keys)}")

        if (isinstance(scope, str) and scope.isdigit()) or isinstance(scope, float):
            scope = int(scope)
        elif isinstance(scope, bool):
            scope = 1 if scope else None
        elif not isinstance(scope, int):
            scope = None

        if isinstance(feat_of_interest, str):
            feat_of_interest = {
                "nc": FeaturesOfInterest.NOUN_CHUNKS,
                "ne": FeaturesOfInterest.NAMED_ENTITIES,
                "both": FeaturesOfInterest.BOTH
            }.get(feat_of_interest.lower(), FeaturesOfInterest.NAMED_ENTITIES)

        self.pseudo_negations = ts["pseudo_negations"]
        self.preceding_negations = ts["preceding_negations"]
        self.following_negations = ts["following_negations"]
        self.termination = ts["termination"]

        self.nlp = nlp
        self.name = name
        self.feature_types = feat_types
        self.extension_name = extension_name
        self.features_of_interest = feat_of_interest
        self.scope = scope
        self.language = language if language is not None else (nlp.lang if nlp.lang is not None else "en")
        self.chunk_prefix = list(nlp.tokenizer.pipe(chunk_prefix))
        self.build_patterns()

    def build_patterns(self):
        # efficiently build spaCy matcher patterns
        self.matcher = PhraseMatcher(self.nlp.vocab, attr="LOWER")

        self.pseudo_patterns = list(self.nlp.tokenizer.pipe(self.pseudo_negations))
        self.matcher.add("pseudo", None, *self.pseudo_patterns)

        self.preceding_patterns = list(
            self.nlp.tokenizer.pipe(self.preceding_negations)
        )
        self.matcher.add("Preceding", None, *self.preceding_patterns)

        self.following_patterns = list(
            self.nlp.tokenizer.pipe(self.following_negations)
        )
        self.matcher.add("Following", None, *self.following_patterns)

        self.termination_patterns = list(self.nlp.tokenizer.pipe(self.termination))
        self.matcher.add("Termination", None, *self.termination_patterns)

    def process_negations(self, doc: Doc):
        """
        Find negations in doc and clean candidate negations to remove pseudo negations

        Parameters
        ----------
        doc: object
            spaCy Doc object

        Returns
        -------
        preceding: list
            list of tuples for preceding negations
        following: list
            list of tuples for following negations
        terminating: list
            list of tuples of terminating phrases

        """
        ###
        # does not work properly in spacy 2.1.8. Will incorporate after 2.2.
        # Relying on user to use NER in meantime
        # see https://github.com/jenojp/negspacy/issues/7
        ###
        # if not doc.is_nered:
        #     raise ValueError(
        #         "Negations are evaluated for Named Entities found in text. "
        #         "Your SpaCy pipeline does not included Named Entity resolution. "
        #         "Please ensure it is enabled or choose a different language model that includes it."
        #     )
        preceding = list()
        following = list()
        terminating = list()

        matches = self.matcher(doc)
        pseudo = [
            (match_id, start, end)
            for match_id, start, end in matches
            if self.nlp.vocab.strings[match_id] == "pseudo"
        ]
        _pseudo_spans = [doc[p[1]:p[2]] for p in pseudo]

        for match_id, start, end in matches:
            if self.nlp.vocab.strings[match_id] == "pseudo":
                continue
            pseudo_flag = False
            _spans = SpanGroup(doc, spans=_pseudo_spans.copy() + [doc[start:end]])
            if _spans.has_overlap:
                pseudo_flag = True
            if not pseudo_flag:
                if self.nlp.vocab.strings[match_id] == "Preceding":
                    preceding.append((match_id, start, end))
                elif self.nlp.vocab.strings[match_id] == "Following":
                    following.append((match_id, start, end))
                elif self.nlp.vocab.strings[match_id] == "Termination":
                    terminating.append((match_id, start, end))
                else:
                    logging.warnings(
                        f"phrase {doc[start:end].text} not in one of the expected matcher types."
                    )
        return preceding, following, terminating

    def termination_boundaries(self, doc, terminating):
        """
        Create sub sentences based on terminations found in text.

        Parameters
        ----------
        doc: object
            spaCy Doc object
        terminating: list
            list of tuples with (match_id, start, end)

        returns
        -------
        boundaries: list
            list of tuples with (start, end) of spans

        """
        sent_starts = [sent.start for sent in doc.sents]
        terminating_starts = [t[1] for t in terminating]
        starts = sent_starts + terminating_starts + [len(doc)]
        starts.sort()
        boundaries = list()
        index = 0
        for i, start in enumerate(starts):
            if not i == 0:
                boundaries.append((index, start))
            index = start
        return boundaries

    def negex(self, doc: Doc):
        """
        Negates entities of interest

        Parameters
        ----------
        doc: object
            spaCy Doc object

        """
        # ToDo: the scope check relies right now on a dependency parse, is this the best way? Maybe constituency parse?
        #  also this won't probably work well with NERs
        #  also I need to account for conjunctions!
        preceding, following, terminating = self.process_negations(doc)
        boundaries = self.termination_boundaries(doc, terminating)
        for b in boundaries:
            sub_preceding = [i for i in preceding if b[0] <= i[1] < b[1]]
            sub_following = [i for i in following if b[0] <= i[1] < b[1]]

            for foi in self.features_of_interest:
                for ft in getattr(doc[b[0]: b[1]], foi):
                    if self.feature_types:
                        if ft.label_ not in self.feature_types:
                            continue
                    if self.chunk_prefix:
                        if self.scope is not None and self.scope > 0:
                            if set(f.text.lower() for f in islice(ft.root.lefts, self.scope)).intersection(
                                    cp.text.lower() for cp in self.chunk_prefix):
                                ft._.set(self.extension_name, True)
                                continue
                        elif any(
                                ft.text.lower().startswith(c.text.lower())
                                for c in self.chunk_prefix
                        ):
                            ft._.set(self.extension_name, True)
                            continue
                    # sorts by biggest span; i.e. token count - most first
                    sorted_sub_preceding = sorted(sub_preceding, key=lambda s: s[2] - s[1], reverse=True)
                    if any(pre[1] < ft.start for pre in sorted_sub_preceding):
                        if self.scope is not None and self.scope > 0:
                            _span_group = self._get_span_groups_right(doc, ft, sorted_sub_preceding[0])
                            if not _span_group.has_overlap:
                                continue
                        ft._.set(self.extension_name, True)
                        continue
                    sorted_sub_following = sorted(sub_following, key=lambda s: s[2] - s[1], reverse=True)
                    if any(fol[2] > ft.end for fol in sorted_sub_following):
                        if self.scope is not None and self.scope > 0:
                            _span_group = self._get_span_groups_left(doc, ft, sorted_sub_following[0])
                            if not _span_group.has_overlap:
                                continue
                        ft._.set(self.extension_name, True)
                        continue
        return doc

    def _get_span_groups_right(self, doc, feature, negation_span, is_root=False, prev_root=None):
        # if scope is set, checks whether the dependents of the negation ('_rights') are within scope
        #  and only negates the ones that are
        _negation_root = doc[negation_span[1]:negation_span[2]].root
        _right_children = [c for c in _negation_root.rights if c.dep_ in RightsDependencyLabels.labels(self.language)]
        if _right_children:
            return SpanGroup(doc, spans=[doc[t.i:t.i + 1] for t in _right_children[:self.scope]] + [feature])
        elif _negation_root.head:
            if is_root or _negation_root == prev_root:
                return SpanGroup(doc, spans=[])
            return self._get_span_groups_right(
                doc, feature, (-1, _negation_root.head.i, _negation_root.head.i + 1),
                _negation_root.head.dep == "ROOT", prev_root=_negation_root)

    def _get_span_groups_left(self, doc, feature, negation_span, is_root=False, prev_root=None):
        _negation_root = doc[negation_span[1]:negation_span[2]].root
        _left_children = [c for c in _negation_root.lefts if c.dep_ in LeftsDependencyLabels.labels(self.language)]
        if _left_children:
            return SpanGroup(doc, spans=[doc[t.i:t.i + 1] for t in _left_children[:self.scope]] + [feature])
        elif _negation_root.head:
            if is_root or _negation_root == prev_root:
                return SpanGroup(doc, spans=[])
            return self._get_span_groups_left(
                doc, feature, (-1, _negation_root.head.i, _negation_root.head.i + 1),
                _negation_root.head.dep == "ROOT", prev_root=_negation_root)

    def __call__(self, doc):
        return self.negex(doc)
