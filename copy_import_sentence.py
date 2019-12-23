import sys
import os

f = open('main.py','r+',encoding='UTF-8')
contents = f.readlines()

import_content = []
for line in contents:
    if(line.startswith('from') or line.startswith('import')):
        import_content.append(line)
        
f.close()
f = open('prune_main.py','r+',encoding='UTF-8')
contents = f.readlines()
contents = import_content + contents
#f.writelines(contents)
f.close()  #如果不关闭，好像用r+，会出现重复了一整次，如果是w+，则contests就没有readlines到东西，只有import的语句

f = open('prune_main.py','w+',encoding='UTF-8')
f.writelines(contents)

f.close()
