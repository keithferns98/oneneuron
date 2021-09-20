l=[1,2,3,4,5,"Keith",['Ferns']]
for i in l:
    if type(i)==str:
        print(i)
for j in l:
    if type(j)==list:
        print(j)