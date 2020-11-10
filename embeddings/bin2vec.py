from fasttext import load_model
from pickle import load
import errno

def convert_to_vec(input_path, output_path, option):
    
    if option == 'bin':
        model = load_model(input_path)
        words = model.get_words()
        file = open(output_path, "w")
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
        file.close()
        print(cnt)
    
    elif option == 'pkl':
        cnt = 0
        with open(input_path, 'rb') as file2:
            model = load(file2)
        file2.close()
        words = model.keys()
        file = open(output_path, "w")
        for w in words:
            v = model[w]
            vstr = ""
            for value in v:
                vstr += " " + str(value)
            try:
                row = w + vstr + "\n"
                file.write(row)
                cnt += 1
            except IOError as e:
                if e.errno == errno.EPIPE:
                    pass
        print(cnt)

    
if __name__ == '__main__':

    input_path = "/home/eastwind/word-embeddings/LSA/TechDofication.LSA.mr.cleaned.d100.pkl"
    output_path = "/home/eastwind/word-embeddings/LSA/TechDofication.LSA.mr.cleaned.d100.vec"
    option = "pkl"
    
    convert_to_vec(input_path, output_path, option)

