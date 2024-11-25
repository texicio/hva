from requests import get
from pyquery import PyQuery
from transformers import pipeline

from lib.features.lexical import Lexical
import stanza

page = "https://www.theverge.com/24169086/sonos-ace-headphones-review"
pq = PyQuery(get(page, timeout=2).text)
text = pq("main p").text()
print(text.split("."))

model = pipeline(
    "text-classification", model="j-hartmann/emotion-english-distilroberta-base"
)
nlp = stanza.Pipeline("en")
doc = nlp(text.strip())
# emotions = model(text.split("."))


lexical = Lexical(text=text)
print([doc.print_dependencies() for doc in doc.sentences])
print(lexical.model_dump())
