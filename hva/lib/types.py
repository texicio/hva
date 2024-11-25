from string import punctuation
from functools import cached_property
from pydantic import BaseModel, computed_field

import numpy as np
import nltk
from textblob import TextBlob, WordList, Sentence
from nltk import ngrams
from transformers import pipeline

emotion = pipeline(
    "sentiment-analysis", top_k=None, model="michellejieli/emotion_text_classifier"
)


class Text(BaseModel):
    """_summary_: Base Model for Text Analysis"""

    text: str
    tag_grams: int = 3

    @staticmethod
    def clean_text(text: str) -> str:
        """_summary_"""
        text_ = text.strip().split()
        text = " ".join([i.strip().lower() for i in text_])
        return text

    def make_taggram(self, n: int) -> dict:
        """_summary_"""
        _tags = ["_".join(i) for i in ngrams(self.tags, n)]
        return {i: _tags.count(i) for i in set(_tags)}

    def _entropy(self, text: str) -> float:
        """_summary_"""
        return np.sum(
            [
                -i / len(text) * np.log2(i / len(text))
                for i in np.unique(list(text), return_counts=True)[1]
            ]
        )

    @cached_property
    def blob(self) -> TextBlob:
        """_summary_"""
        return TextBlob(self.clean_text(self.text)).correct()

    @cached_property
    def words(self) -> WordList:
        """_summary_"""
        return self.blob.words

    @cached_property
    def sentences(self) -> list[Sentence]:
        """_summary_"""
        return self.blob.sentences

    @cached_property
    def nouns(self) -> WordList:
        """_summary_"""
        return self.blob.noun_phrases

    @cached_property
    def text_norm(self) -> str:
        """_summary_"""
        return self.clean_text(self.text.strip(punctuation))

    @cached_property
    def tagged_tokens(self) -> list[tuple[str, str]]:
        """_summary_"""
        return self.blob.tags

    @cached_property
    def tags(self) -> list[str]:
        """_summary_"""
        return [i[-1].lower() for i in self.tagged_tokens]

    @cached_property
    def entities(self) -> list[dict]:
        """_summary_"""
        entities = nltk.ne_chunk(self.tagged_tokens)
        return list(
            filter(
                lambda e: isinstance(e["entity"], nltk.Tree),
                map(
                    lambda entity: dict(pos=entities.index(entity), entity=entity),
                    entities,
                ),
            )
        )

    @cached_property
    def _emotions(self) -> list:
        """_summary_"""
        # todo: get scores for all emotions
        return [emotion(str(i))[0] for i in self.sentences]

    @computed_field
    @cached_property
    def sentiment(self) -> tuple:
        """_summary_"""
        return self.blob.sentiment

    @computed_field
    @cached_property
    def polarity(self) -> float:
        """_summary_"""
        return self.sentiment[0]

    @computed_field
    @cached_property
    def subjectivity(self) -> float:
        """_summary_"""
        return self.sentiment[1]

    @computed_field
    @cached_property
    def token_count(self) -> int:
        """_summary_"""
        return len(self.words)

    @computed_field
    @cached_property
    def character_count(self) -> int:
        """_summary_"""
        return len(self.text.replace(" ", ""))

    @computed_field
    @cached_property
    def unitags(self) -> dict:
        """_summary_"""
        return self.make_taggram(1)

    @computed_field
    @cached_property
    def bitags(self) -> dict:
        """_summary_"""
        return self.make_taggram(2)

    @computed_field
    @cached_property
    def tritags(self) -> dict:
        """_summary_"""
        return self.make_taggram(3)

    @computed_field
    @cached_property
    def entropy(self) -> float:
        """_summary_"""
        return self._entropy(self.text)

    @computed_field
    @cached_property
    def avg_entropy_change(self) -> float:
        """_summary_"""
        diff = np.diff([self._entropy(str(i)) for i in self.sentences])
        return sum(diff) / len(self.sentences)

    @computed_field
    @cached_property
    def avg_polarity_change(self) -> float:
        """_summary_"""
        diff = np.diff([i.polarity for i in self.sentences])
        return sum(diff) / len(self.sentences)

    @computed_field
    @cached_property
    def avg_subjectivity_change(self) -> float:
        """_summary_"""
        diff = np.diff([i.subjectivity for i in self.sentences])
        return sum(diff) / len(self.sentences)

    @computed_field
    @cached_property
    def unique_words_count(self) -> int:
        """_summary_"""
        return len(set([i.lower() for i in self.text_norm.split()]))

    @computed_field
    @cached_property
    def avg_token_length(self) -> float:
        """_summary_"""
        return np.mean([len(i) for i in self.words])
