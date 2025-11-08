import requests
from bs4 import BeautifulSoup
import re
from lxml import etree
import json
import os
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
rel_path = "drug.txt"
abs_file_path = os.path.join(script_dir, rel_path)
file1 = open(abs_file_path, 'r')
drugs=[]
for i in range(708):
    line = file1.readline()
    if not line:
        break
    drugs.append(line.strip())
 
file1.close()


drugs2=[]
for dr in tqdm(drugs):
    response = requests.get('https://go.drugbank.com/structures/small_molecule_drugs/'+dr+'.smile', headers=headers)
    soup = BeautifulSoup(response.content, 'html.parser')
    
    if str(soup) == "":
         response = requests.get('https://alpha.dmi.unict.it/dtweb/structure.php?drug_id='+dr+'&type=smiles', headers=headers)
         soup = BeautifulSoup(response.content, 'html.parser')
         if str(soup) == "":
             print("F ME",dr)
    drugs2.append(str(soup))
drug_smile=dict(zip(drugs,drugs2))
print( any(x is None for x in drugs2))
print(drug_smile)
with open('drugs_smiles.txt', 'w') as file:
     file.write(json.dumps(drug_smile))
     file.close()