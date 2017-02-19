import string

def build_dict():
    basic_english_file = "hidestyle/data/th_en_BE.dat"
    words = {}
    with open(basic_english_file) as f:
        lines = f.readlines()
        skip = 0
        prev_word = ''
        for l in lines[1:]:
            if skip > 0:
                words[prev_word] = l.strip()
                skip -= 1
                continue
            prev_word, count = l.split('|')
            skip += int(count)
            
    for k,v in list(words.items()):
        words[k] = words[k].split(',')[0]
        if any([i in words[k] for i in list(string.punctuation)]):
            del words[k]

def replace_words(words_dict, text, targets):
    for w in targets:
        if w in words_dict:
            text = text.replace(w, words_dict[w])
    return text