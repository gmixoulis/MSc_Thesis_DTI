import requests
from bs4 import BeautifulSoup
import re
from lxml import etree
import json
import os
import ast
from collections import defaultdict
import pandas as pd
from tqdm import tqdm
headers = {
    'authority': 'www.scorebing.com',
    'pragma': 'no-cache',
    'cache-control': 'no-cache',
    'upgrade-insecure-requests': '1',
    'user-agent': 'Mozilla/5.0 (Windows NT 10.0; WOW64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/81.0.4044.138 Safari/537.36',
    'accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,image/apng,*/*;q=0.8,application/signed-exchange;v=b3;q=0.9',
    'sec-fetch-site': 'none',
    'sec-fetch-mode': 'navigate',
    'sec-fetch-user': '?1',
    'sec-fetch-dest': 'document',
    'accept-language': 'en-US,en;q=0.9',
    
}



prot_path='C://Users//gmixo//Desktop//data//'

script_dir = os.path.dirname(prot_path) #<-- absolute dir the script is in
rel_path = "protein.txt"
abs_file_path = os.path.join(script_dir, rel_path)

file1 = open(abs_file_path, 'r')
proteins=[]
for i in range(1512):
    line = file1.readline()
    if not line:
        break
    proteins.append(line.strip())
 
file1.close()

print(len(proteins))
proteins2=[]
for prot in tqdm(proteins):
    response = requests.get('https://www.uniprot.org/uniprot/'+prot+'.fasta', headers=headers)
    soup = BeautifulSoup(response.content, 'html.parser')
    #print(str(soup))
    if str(soup) == "":
        print (prot)
    s=re.sub(r'&gt;','>',str(soup))
    s=re.sub(r'(?<!\n)\n(?!\n)', ' \n', s)
    if(not s.startswith(">")):
        print(prot)

    #print(s)
    #proteins2.append(s)
    #d=len(str(soup).rsplit("\n"))
    #dds=str(soup).rsplit("\n")
    #protein=""
    #for i in range(0,d):
    #    protein=protein+dds[i]
    proteins2.append(s)
print(len(proteins2))
print(pd.DataFrame(proteins2).shape)
    
         
protein_sequence=list(zip(proteins,proteins2))
print(protein_sequence)
with open('file111.txt', 'w+') as file:
     file.write(json.dumps(protein_sequence))
     
     file.close()
'''
with open('file111.txt') as f:
    data = f.read()
    protein_dict = json.loads(data)
    #print(protein_dict.values().count(">"))
    count =0
    for i in protein_dict.keys():
        
            count = count+1
            #print(i)
    print(count)
'''