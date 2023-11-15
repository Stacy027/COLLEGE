from bs4 import BeautifulSoup
import sys
from urllib import parse
import os
from multiprocessing import Pool

input_folder = "../pretrain_data/output"

file_list = []
for path, _, filenames in os.walk(input_folder):
    for filename in filenames:
        file_list.append(os.path.join(path, filename))
print(len(file_list))

def run_proc(idx, n, file_list):
    for i in range(len(file_list)):
        if i % n == idx:
            input_name = file_list[i]
            print('{}: {}'.format(i, input_name))
            target = input_name.replace('pretrain_data/output', "pretrain_txt/output")
            folder = '/'.join(target.split('/')[:-1])
            if not os.path.exists(folder):
                os.makedirs(folder)

            soup = BeautifulSoup(open(input_name, encoding='utf-8'), features="html5lib")
            docs = soup.find_all('doc')

            fout = open(target, 'w', encoding='utf-8')

            for doc in docs:
                content = doc.get_text(" sepsepsep ")
                while content[0] == "\n":
                    content = content[1:]
                content = [x.strip() for x in content.split("\n")]
                content = "".join(content[1:])

                lookup = []
                for x in doc.find_all("a"):
                    if x.get('href') is not None:
                        lookup.append((x.get_text().strip(), parse.unquote(x.get('href'))))
                # lookup = [(x.get_text().strip(), parse.unquote(x.get('href'))) for x in doc.find_all("a")]
                lookup = "[_end_]".join(["[_map_]".join(x) for x in lookup])
                fout.write(content+"[_end_]"+lookup+"\n")

            fout.close()


n = int(sys.argv[1])
p = Pool(n)
for i in range(n):
    p.apply_async(run_proc, args=(i, n, file_list))
p.close()
p.join()