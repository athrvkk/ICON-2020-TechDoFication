from fasttext import load_model
import errno
import pickle

def bin2vec(input_file, output_file):

    model = load_model(input_file)
    file = open(output_file, "w",encoding='utf')
    words = model.get_words()
    print('Input Vocab:\t',str(len(words)), "\nModel Dimensions: ",str(model.get_dimension()))
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
    print('Words processed: ',cnt)


def pickel2vec(input_file, output_file):


    file1 = open(output_file, "w",encoding= 'utf-8')
    with open(input_file,'rb') as file2:
        model = pickle.load(file2)

    words = model.keys()
    type(words)
    print("Input Vocab:",len(words))

    cnt = 0

    for w in words:

        v = model[w]
        vstr = ""

        for value in v:
            vstr += " " + str(value)

        try:
            row = w + vstr + "\n"
            file1.write(row)
            cnt += 1

        except Exception as e:
            print('Exception: ',e)
            # if e.errno == errno.EPIPE:
            # pass

    print('Words processed: ',cnt)


if __name__ == '__main__':

    input_file = "/home/eastwind/word-embeddings/fasttext/TechDofication.hi.raw.complete.ft.skipgram.d300.bin"
    output_file = "/home/eastwind/word-embeddings/fasttext/TechDofication.hi.raw.complete.ft.skipgram.d300.vec"

    bin2vec(input_file,output_file)
