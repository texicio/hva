import re
from string import ascii_letters, punctuation
from functools import cached_property

from pydantic import computed_field
from hva.lib.types import Text


class Lexical(Text):
    """_summary_: Lexical analysis of text"""

    @cached_property
    def vowels(self) -> str:
        """_summary_: Vowels in the English alphabet"""
        return "aeiouAEIOU"

    @cached_property
    def consonants(self) -> str:
        """_summary_: Consonants in the English alphabet"""
        return "".join([i for i in ascii_letters if i not in self.vowels])

    @cached_property
    def punctuations(self) -> str:
        """_summary_: Punctuation marks"""
        return punctuation

    @computed_field
    @cached_property
    def vowel_count(self) -> int:
        """_summary_: Number of vowels in the text"""
        return len(re.findall(f"[{self.vowels}]", self.text))

    @computed_field
    @cached_property
    def consonant_count(self) -> int:
        """_summary_: Number of consonants in the text"""
        return len(re.findall(f"[{self.consonants}]", self.text))

    @computed_field
    @cached_property
    def punctuation_count(self) -> int:
        """_summary_: Number of punctuation marks in the text"""
        return len(re.sub(f"[^{punctuation}]", "", self.text))

    @computed_field
    @cached_property
    def unique_punctuation_count(self) -> int:
        """_summary_: Number of unique punctuation marks in the text"""
        return len(set(re.sub(f"[^{punctuation}]", "", self.text)))

    @computed_field
    @cached_property
    def word_count(self) -> int:
        """_summary_: Number of words in the text"""
        return len(self.words)

    @computed_field
    @cached_property
    def unique_word_count(self) -> int:
        """_summary_: Number of unique words in the text"""
        return len(set(self.words))

    @computed_field
    @cached_property
    def sentence_count(self) -> int:
        """_summary_: Number of sentences in the text"""
        return len(self.sentences)

    @computed_field
    @cached_property
    def noun_count(self) -> int:
        """_summary_: Number of nouns in the text"""
        return len(self.nouns)

    @computed_field
    @cached_property
    def digit_count(self) -> int:
        """_summary_: Number of digits in the text"""
        return len(re.findall("[^0-9]", self.text))

    @computed_field
    @cached_property
    def break_count(self) -> int:
        """_summary_: Number of line breaks in the text"""
        return len(re.findall(",", self.text))

    @computed_field
    @cached_property
    def hyphenated_word_count(self) -> int:
        """_summary_: Number of hyphenated words in the text"""
        return len(re.findall(r"\w+-\w+", self.text))

    @computed_field
    @cached_property
    def hyphenated_phrase_count(self) -> int:
        """_summary_: Number of hyphenated phrases in the text"""
        return len(re.findall(r"\w+ - \w+", self.text))

    @computed_field
    @cached_property
    def capitalized_word_count(self) -> int:
        """_summary_: Number of capitalized words in the text"""
        return len(re.findall(r"[A-Z][a-z]+", self.text))

    @computed_field
    @cached_property
    def capitalized_count(self) -> int:
        """_summary_: Number of capitalized characters in the text"""
        return len(re.findall(r"[A-Z]", self.text))

    @computed_field
    @cached_property
    def lowercase_count(self) -> int:
        """_summary_: Number of lowercase characters in the text"""
        return len(re.findall(r"[a-z]", self.text))

    @computed_field
    @cached_property
    def decimal_count(self) -> int:
        """_summary_: Number of decimal numbers in the text"""
        return len(re.findall(r"\d+\.\d+", self.text))

    @computed_field
    @cached_property
    def avg_words_per_sentence(self) -> float:
        """_summary_: Average number of words per sentence"""
        return self.word_count / self.sentence_count

    @computed_field
    @cached_property
    def vowel_density(self) -> float:
        """_summary_: Proportion of vowels in the text"""
        return self.vowel_count / self.character_count

    @computed_field
    @cached_property
    def consonant_density(self) -> float:
        """_summary_: Proportion of consonants in the text"""
        return self.consonant_count / self.character_count

    @computed_field
    @cached_property
    def punctuation_density(self) -> float:
        """_summary_: Proportion of punctuation marks in the text"""
        return self.punctuation_count / self.character_count

    @computed_field
    @cached_property
    def noun_density(self) -> float:
        """_summary_: Proportion of nouns in the text"""
        return self.noun_count / self.word_count

    @computed_field
    @cached_property
    def digit_density(self) -> float:
        """_summary_: Proportion of digits in the text"""
        return self.digit_count / self.character_count

    @computed_field
    @cached_property
    def quoted_count(self) -> int:
        """_summary_: Number of quoted phrases in the text"""
        return len(re.findall(r"\".*?\"", self.text))

    @computed_field
    @cached_property
    def break_density(self) -> float:
        """_summary_: Proportion of line breaks in the text"""
        return self.break_count / self.character_count

    @computed_field
    @cached_property
    def capitalization_density(self) -> float:
        """_summary_: Proportion of capitalized characters in the text"""
        return self.capitalized_count / self.character_count

    @computed_field
    @cached_property
    def vowel_following_vowels(self) -> int:
        """_summary_: Number of vowels following vowels in the text"""
        return len(re.findall(f"[{self.vowels}][{self.vowels}]", self.text))

    @computed_field
    @cached_property
    def consonant_following_consonants(self) -> int:
        """_summary_: Number of consonants following consonants in the text"""
        return len(re.findall(f"[{self.consonants}][{self.consonants}]", self.text))

    @computed_field
    @cached_property
    def vowel_following_consonants(self) -> int:
        """_summary_: Number of vowels following consonants in the text"""
        return len(re.findall(f"[{self.consonants}][{self.vowels}]", self.text))

    @computed_field
    @cached_property
    def consonant_following_vowels(self) -> int:
        """_summary_: Number of consonants following vowels in the text"""
        return len(re.findall(f"[{self.vowels}][{self.consonants}]", self.text))

    @computed_field
    @cached_property
    def digit_following_digits(self) -> int:
        """_summary_: Number of digits following digits in the text"""
        return len(re.findall(r"\d\d", self.text))

    @computed_field
    @cached_property
    def decimal_following_digits(self) -> int:
        """_summary_: Number of decimal numbers following digits in the text"""
        return len(re.findall(r"\d\.\d", self.text))

    @computed_field
    @cached_property
    def hyphenated_word_following_hyphenated_words(self) -> int:
        """_summary_: Number of hyphenated words following hyphenated words in the text"""
        return len(re.findall(r"\w+-\w+-\w+", self.text))

    @computed_field
    @cached_property
    def digit_following_vowel(self) -> int:
        """_summary_: Number of digits following vowels in the text"""
        return len(re.findall(f"[{self.vowels}]\d", self.text))

    @computed_field
    @cached_property
    def digit_following_consonant(self) -> int:
        """_summary_: Number of digits following consonants in the text"""
        return len(re.findall(f"[{self.consonants}]\d", self.text))

    @computed_field
    @cached_property
    def digit_following_punctuation(self) -> int:
        """_summary_: Number of digits following punctuation marks in the text"""
        return len(re.findall(f"[{punctuation}]\d", self.text))

    @computed_field
    @cached_property
    def digit_following_break(self) -> int:
        """_summary_: Number of digits following line breaks in the text"""
        return len(re.findall(",\d", self.text))

    @computed_field
    @cached_property
    def vowel_following_digit(self) -> int:
        """_summary_: Number of vowels following digits in the text"""
        return len(re.findall(r"\d[{self.vowels}]", self.text))

    @computed_field
    @cached_property
    def consonant_following_digit(self) -> int:
        """_summary_: Number of consonants following digits in the text"""
        return len(re.findall(r"\d[{self.consonants}]", self.text))

    @computed_field
    @cached_property
    def vowel_following_punctuation(self) -> int:
        """_summary_: Number of vowels following punctuation marks in the text"""
        return len(re.findall(f"[{punctuation}][{self.vowels}]", self.text))

    @computed_field
    @cached_property
    def consonant_following_punctuation(self) -> int:
        """_summary_: Number of consonants following punctuation marks in the text"""
        return len(re.findall(f"[{punctuation}][{self.consonants}]", self.text))
