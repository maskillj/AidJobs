from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from datetime import datetime
import time
from bs4 import BeautifulSoup
import torch
import selenium
import pandas as pd
import json
import numpy as np
from transformers import AutoModelForQuestionAnswering, AutoTokenizer, pipeline

model_name = "deepset/roberta-base-squad2"
nlp = pipeline('question-answering', model=model_name, tokenizer=model_name)
model = AutoModelForQuestionAnswering.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)
language_name = ['Abaza','Abkhaz','Acehnese','Adyghe','Afar','Afrikaans','Akan','Aklan','Albanian','Algonquin','Altay','Amharic','Angika','Arabic','Aramaic','Armenian','Assamese','Ateso','Awadhi','Aymara','Azerbaijani','Balinese','Balochi','Balti','Bambara','Bangla','Banjar','Banyumasan','Bashkir','Basque','Batak Toba','Beja','Belarusian','Bengali','Bhili','Bhojpuri','Bikol','Bosnian','Brahui','Buginese','Bukusu','Bulgarian','Burmese','Buryat','Catalan','Cebuano','Chaouia','Chavacano','Chechen','Chhattisgarhi','Chichewa','Chinese','Chittagonian','Chuvash','Comorian','Creole','Croatian','Czech','Danish','Dargin','Dari','Daur','Dhivehi','Dida','Dioula','Dogri','Dutch','Dzongkha','Eastern Yugur','Egyptian','English','Erzya','Estonian','Even','Ewe','Fijian','Filipino','Finnish','Fon','French','Fula','Fur','Ga','Gayo','Georgian','German','Gilaki','Gilbertese','Gondi','Greek','Guarani','Gujarati','Gusii','Hausa','Hebrew','Hiligaynon','Hindi','Hmong','Ho','Hungarian','Iban','Ibanag','Ibibio','Icelandic','Igbo','Ikalanga','Ilokano','Indonesian','Ingush','Irish','Isan','Italian','Japanese','Jarai','Javanese','Kabardian','Kabyle','Kalmyk','Kankanaey','Kannada','Kapampangan','Karachay-Balkar','Karakalpak','Kashmiri','Kazakh','Khandeshi','Khasi','Khmer','Khowar','Kinaray-a','Kinyarwanda','Kirombo','Kirundi','Kivunjo','Komi','Kongo','Konkani','Korean','Korku','Koya','Kumaoni','Kumyk','Kurdish','Kurukh','Kwanyama','Kyrgyz','Lao','Latin','Latvian','Lezgian','Lingala','Lithuanian','Lori','Lozi','Luganda','Lunda','Lusoga','Macedonian','Magahi','Maguindanao','Maithili','Makassar','Makhuwa','Makhuwa-Meetto','Malagasy','Malay','Malayalam','Maltese','Malvi','Mam','Mandinka','Mapudungun','Maranao','Marathi','Mari','Masaba','Masbateno','Mazandarani','Meitei','Minangkabau','Moksha','Mon','Mongolian','Montenegrin','Nahuatl','Nama','Ndonga','Nepal Bhasa','Nepali','Norwegian','Nuer','Oriya','Oromo','Ossetic','Pangasinan','Papiamento','Pashto','Persian/Farsi','Polish','Portuguese','Pothohari','Punjabi','Qashqai','Quechua','Rajasthani','Romani','Romanian','Russian','Sakha','Samoan','Sango','Sanskrit','Santali','Saraiki','Sardinian','Saurashtra','Serbian','Serbo-Croatian','Shina','Shona','Sicilian','Sidamo','Sign (Language)','Silesian','Silt\'e','Sindhi','Sinhalese','Slovak','Slovenian','Soddo','Somali','Sora','Sotho','Spanish','Sundanese','Supyire','Surigaonon','Surinamese','Susu','Swahili','Swati','Swedish','Syriac','Tagalog','Tahitian','Tajik','Talysh','Tamil','Tarifit','Tashelhiyt','Tatar','Tausug','Telugu','Tetum','Thai','Tibetan','Tigre','Tigrinya','Tiv','Tok Pisin','Tonga','Tongan','Tshiluba','Tsonga','Tswana','Tuareg','Tulu','Tumbuka','Turkish','Turkmen','Tuvaluan','Tuvan','Udmurt','Ukrainian','Urdu','Uyghur','Uzbek','Venda','Vietnamese','Visayan','Welsh','Wolof','Xhosa','Yiddish','Yoruba','Yucatec Maya','Zapotec','Zazaki','Zoque','Zulu']
language_code = [23,24,25,26,27,28,29,30,31,32,33,35,36,37,38,39,40,41,43,44,45,46,47,48,49,57,50,51,52,53,54,55,56,2,58,59,60,61,62,63,64,65,66,67,314,68,70,69,71,72,74,73,75,76,77,115,78,79,5,81,311,83,84,85,86,87,88,89,90,91,92,93,94,95,96,97,98,99,100,101,102,103,104,105,106,107,109,108,110,111,112,113,114,116,117,118,119,120,121,122,124,125,123,126,127,128,129,130,131,132,133,134,135,136,137,138,139,140,141,142,143,144,145,146,147,148,149,150,151,152,153,154,155,156,157,159,158,160,161,162,163,164,166,165,167,168,169,312,170,171,172,173,177,174,175,176,178,179,180,181,182,183,184,185,186,187,188,189,190,191,192,193,194,195,196,197,198,200,199,201,202,203,204,205,206,207,208,209,210,211,212,213,214,215,216,217,218,82,219,220,221,222,223,224,225,227,226,228,301,229,230,313,231,246,232,233,236,237,235,239,240,241,34,242,243,244,245,247,248,249,250,251,238,253,255,256,257,254,258,259,260,261,262,264,265,266,267,268,269,263,270,271,272,273,274,275,276,277,278,309,279,280,281,282,283,284,285,286,287,288,289,290,291,292,20,294,295,296,297,42,21,299,300,302,308,303,304,305,306,307]

language_dict = {}
for i in range(0,len(language_name)):
    language_dict[language_name[i]]=language_code[i]
region_dict = {"Africa":3,
    "Asia":5,
    "Europe":6,
    "Latin America and the Caribbean":4,
    "Northern America":9,
    "Oceania":7}

def get_info(description):
    outputs = []
    questions = ['What languages are required?','What education is required?',
                'How much experience is required?','Where does this project take place?',
                'What are the duties of this role?','What experiences are desirable?']
    for q in questions: 
        QA_input = {'question': q,
                'context': description}
        outputs.append(nlp(QA_input)['answer'])
    return outputs


def cast_net(name="",job_types="",languages="",region="",minimum_experience="",maximum_experience="",recency="",sectors="",types=""):
    url = "https://www.developmentaid.org/jobs/search?sort=highlighted.desc,postedDate.desc"
    if name != '':
        name = name.replace(' ','%20')
        name = "&organizationName="+name
    if job_types != '':
        job_types=job_types.replace(' ','%20')
        job_types = "&jobTypes="+job_types
    if languages != '':
        language_list = languages.split(",")
        language_codes = []
        for lang in language_list:
            if lang in language_dict.keys():
                lang = str(language_dict[lang])
                language_codes.append(lang)
            else:
                next
        addition = ""
        languages = ''
        if language_codes != []:
            for code in language_codes:
                addition = addition + code + ','
            languages = "&languages="+addition[0:-1]
    if region != '':
        if region in region_dict.keys():
            region = str(region_dict[region])
            region=region.replace(' ','%20')
            region = "&locations="+region
        else:
            region = ""
    if minimum_experience != '':
        minimum_experience=minimum_experience.replace(' ','%20')
        minimum_experience = "&minimumExperience="+minimum_experience
    if maximum_experience != '':
        maximum_experience=maximum_experience.replace(' ','%20')
        maximum_experience = "&maximumExperience="+maximum_experience
    if recency != '':
        recency=recency.replace(' ','%20')
        recency = "&postedDateLessThanDaysAgo="+recency
    if sectors != '':
        sectors=sectors.replace(' ','%20')
        sectors = "&sectors="+sectors
    if types != '':
        types=types.replace(' ','%20')
        types = "&types="+types
    home_page = url+name+job_types+languages+region+minimum_experience+maximum_experience+recency+sectors+types
    browser = webdriver.Chrome()
    browser.get(home_page)
    time.sleep(10)
    anchors = browser.find_elements(By.TAG_NAME, "a")
    urls = [anchor.get_attribute('href') for anchor in anchors]
    browser.quit()
    prefix = "https://www.developmentaid.org/jobs/view/"
    filtered_list = [url for url in urls if url is not None and url.startswith(prefix)]
    unique_list = []
    [unique_list.append(x) for x in filtered_list if x not in unique_list]
    relevant_opportunities = unique_list
    opportunities_dict = {}
    driver = webdriver.Chrome()
    for opportunity in relevant_opportunities:
        driver.get(opportunity)
        try:
            accept_button = WebDriverWait(driver, 1).until(
                EC.element_to_be_clickable((By.ID, 'acceptCookies')))  
            accept_button.click()
        except:
            pass
        content = driver.page_source
        soup = BeautifulSoup(content, "html.parser")
        page_text = soup.get_text(separator=' ', strip=True)
        opportunities_dict[opportunity] = page_text
    driver.quit()
    opportunity_dict = {}
    i = 0
    for opportunity in opportunities_dict.values(): 
        info = get_info(opportunity)
        op = {}
        opportunity_dict[i] = op
        op['Language'] = info[0]
        op['Education'] = info[1]
        op['Seniority'] = info[2]
        op['Location'] = info[3]
        op['Duties'] = info[4]
        op['Experience'] = info[5]
        op['Link to Tender'] = [d for d in opportunity_dict][i]
        i+=1
    df = pd.DataFrame.from_dict(opportunity_dict, orient='index')
    excel_filename = 'Opportunities Summary'
    current_time = datetime.now()
    formatted_timestamp = current_time.strftime("%Y-%m-%d")
    new_string = excel_filename + " " + formatted_timestamp+'.xlsx'
    df.to_excel(new_string, index=False)
    print('Your potential jobs are now available at {}'.format(new_string))