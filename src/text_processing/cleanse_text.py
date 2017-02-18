# spacy lightning tour

# Convert text to lower-case and strip punctuation/symbols from words
def normalize_text(text):
    norm_text = text.lower()

    # Replace breaks with spaces, in case there's broken html in there
    norm_text = norm_text.replace('<br />', ' ')

    # Replace multiple whitespaces with a single space
    norm_text = " ".join(norm_text.split())

    # Remove commas, colons, semicolons. The reader can just deal with it.
    to_remove = [",", ":", ";"]
    norm_text = norm_text.translate({ord(x): '' for x in to_remove})

    # Replace exclamation marks with full stops
    norm_text = norm_text.replace("!", ".")


    # Pad punctuation with spaces on both sides
    for char in ['.', '"', ',', '(', ')', '!', '?', ';', ':']:
        norm_text = norm_text.replace(char, ' ' + char + ' ')

    return norm_text


