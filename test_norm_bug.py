import sys
import os
sys.path.append(os.path.abspath("."))
from indextts.utils.front import TextNormalizer

normalizer = TextNormalizer(enable_glossary=True)
normalizer.load()

test_words = ["coward", "combat", "company", "control", "CO", "Colorado"]
for word in test_words:
    normalized = normalizer.normalize(word)
    print(f"'{word}' -> '{normalized}'")
