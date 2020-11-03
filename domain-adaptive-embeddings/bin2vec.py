from fasttext import load_model
import errno


model = load_model("/home/eastwind/word-embeddings/fasttext/TechDofication.ft.cbow.mr.300.bin")
file = open("/home/eastwind/word-embeddings/fasttext/TechDofication.ft.cbow.mr.300.vec", "w")
words = model.get_words()
print(str(len(words)) + " " + str(model.get_dimension()))
cnt = 0
for w in words:
    v = model.get_word_vector(w)
    vstr = ""
    for vi in v:
        vstr += " " + str(vi)
    try:
        row = w + vstr + "\n"
        file.write(row)
        cnt = cnt + 1
    except IOError as e:
        if e.errno == errno.EPIPE:
            pass
print(cnt)