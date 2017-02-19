import string

words_dict = {}
def build_dict():
    basic_english_file = "../../data/th_en_BE.dat"
    global words_dict
    with open(basic_english_file) as f:
        lines = f.readlines()
        skip = 0
        prev_word = ''
        for l in lines[1:]:
            if skip > 0:
                words_dict[prev_word] = l.strip()
                skip -= 1
                continue
            prev_word, count = l.split('|')
            skip += int(count)
    for k, _ in list(words_dict.items()):
        words_dict[k] = words_dict[k].split(',')[0]
        if any([i in words_dict[k] for i in list(string.punctuation)]):
            del words_dict[k]

def replace_words(text, targets):
    global words_dict
    for w in targets:
        if w in words_dict:
            print("Replacing {0} with {1}".format(w, words_dict[w]))
            text = text.replace(w, words_dict[w])
    return text

build_dict()