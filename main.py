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

country_name = ['Algeria','Angola','Benin','Botswana','Burkina Faso','Burundi','Cameroon', 'Cape Verde','Central African Republic','Chad','Comoros','Congo','Cote d\'Ivoire', 'Dem. Rep. Congo','Djibouti','Egypt','Equatorial Guinea','Eritrea','Eswatini (Swaziland)' ,'Ethiopia','French Southern Territory','Gabon','Gambia','Ghana','Guinea','Guinea-Bissau','Kenya', 'Lesotho','Liberia','Libya','Madagascar','Malawi','Mali','Mauritania','Mauritius','Mayotte', 'Morocco','Mozambique','Namibia','Niger','Nigeria','Reunion','Rwanda','Saint Helena', 'Sao Tome and Principe','Senegal','Seychelles','Sierra Leone','Somalia','Somaliland','South Africa', 'South Sudan','Sudan','Tanzania','Togo','Tunisia','Uganda','Western Sahara','Zambia','Zimbabwe', 'Afghanistan','Armenia','Azerbaijan','Bahrain','Bangladesh', 'Bhutan','Brunei','Cambodia','China','Georgia','Hong Kong','India','Indonesia','Iran','Iraq', 'Israel','Japan','Jordan','Kazakhstan','Kuwait','Kyrgyzstan','Laos','Lebanon','Macao','Malaysia', 'Maldives','Mongolia','Myanmar','Nepal','North Korea','Oman','Pakistan', 'Palestine / West Bank & Gaza','Qatar','Saudi Arabia','Singapore','South Korea','Sri Lanka', 'Syria','Taiwan','Tajikistan','Thailand','Timor-Leste','Turkmenistan','UAE','Uzbekistan', 'Vietnam','Yemen','Philippines','Austria', 'Azores','Belgium','Bulgaria','Canary Islands','Croatia','Cyprus','Czech Republic','Denmark', 'Estonia','Faroe Islands','Finland','France','Germany','Greece','Hungary','Ireland','Italy', 'Latvia','Lithuania','Luxembourg','Malta','Netherlands','Poland','Portugal','Romania','Slovakia', 'Slovenia','Spain','Sweden','Madeira','Aland Islands','Albania','Andorra', 'Belarus','Bosnia and Herzegovina','Channel Islands','Gibraltar','Iceland','Isle of Man','Kosovo', 'Liechtenstein','Moldova','Monaco','Montenegro','North Macedonia','Norway','Russia','Serbia', 'Svalbard','Switzerland','Turkey','UK','Ukraine','Vatican City','San Marino','Anguilla', 'Antigua and Barbuda','Argentina','Aruba','Bahamas','Barbados','Belize','Bolivia','Brazil', 'British Virgin Islands','Caribbean Netherlands','Cayman Islands','Chile','Colombia','Costa Rica', 'Cuba','Curaçao','Dominica','Dominican Republic','Ecuador','El Salvador', 'Falkland Islands','French Guiana','Galapagos','Grenada','Guadeloupe','Guatemala','Guyana','Haiti', 'Honduras','Jamaica','Martinique','Mexico','Nicaragua','Panama','Paraguay','Peru','Puerto Rico', 'Saint Kitts and Nevis','Saint Lucia','Saint Martin','Saint Vincent and the Grenadines', 'Sint Maarten','Suriname','Trinidad and Tobago','Turks and Caicos','Uruguay','US Virgin Islands', 'Venezuela','Montserrat','Bermuda','Canada','Greenland','USA','St. Pierre and Miquelon', 'American Samoa','Australia','Christmas Island','Cocos (Keeling) Islands','Cook Islands', 'Easter Island','Fiji','French Polynesia','Guam','Heard and McDonald Islands','Kiribati', 'Marshall Islands','Micronesia','Nauru','New Caledonia','New Zealand','Niue','Norfolk Island', 'Northern Mariana Islands','Papua New Guinea','Pitcairn','Samoa','Solomon Islands','Tokelau', 'Tonga','Tuvalu','Vanuatu','Wallis and Futuna','Palau']
country_code = [10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,61,28,29,30,31,32,33,34,35,36,37,38,39,40,41,42,43,44,45,46,47,48,49,50,51,52,53,54,55,56,57,300,58,59,60,62,63,64,65,66,67,68,136,137,138,139,140,141,142,143,144,145,146,147,148,149,150,151,152,153,154,155,156,157,158,159,160,161,162,163,164,165,166,167,168,170,171,172,173,174,175,176,177,178,179,180,181,182,183,184,169,199,200,201,202,203,204,205,206,207,208,209,210,211,212,213,214,215,216,217,218,219,220,221,222,223,224,225,226,227,228,273,261,262,263,264,265,266,267,268,269,270,271,274,275,276,272,277,278,280,281,282,283,229,284,285,279,73,74,75,76,77,78,79,80,81,82,297,83,84,85,86,87,298,88,89,90,91,92,93,94,95,96,97,98,99,100,101,102,103,106,107,108,109,110,111,112,113,114,299,115,116,117,118,119,120,104,292,294,296,293,295,230,231,232,233,234,235,236,237,238,239,240,241,242,243,244,245,246,247,248,250,251,252,253,254,255,256,257,258,249]
country_dict={}
for i in range(len(country_name)):
    country_dict[country_name[i]]=country_code[i]

region_dict = {"Africa":3,"Asia":5,"Europe":6,"Latin America and the Caribbean":4,"Northern America":9,"Oceania":7}

sector_name = ['Administration','Advocacy','Agriculture','Air & Aviation','Anti-Corruption','Architecture','Audit','Banking','Border Management','Civil Engineering','Civil Society & NGOs','Conflict','Corporate Social Responsibility','Culture','Decentralization & Local Development','Democratization','Design','Disaster Reduction','Education','Electrical Engineering','Energy','Environment & NRM','Finance & Accounting','Fisheries & Aquaculture','Food Processing & Safety','Food Security','Fundraising','Furniture & Office Supplies','Gender','Grants & Grant Schemes','Health','Heating','Human Resources','Human Rights','Humanitarian Aid & Emergency','Industry, Commerce & Services','Information & Communication Technology','Inst. Devt. & Cap. building','Justice Reform','Laboratory & Measurement','Labour Market & Employment','Land & Erosion & Soil','Law','Livestock (incl. animal/bird production & health)','Logistics','Macro-Econ. & Public Finance','Mapping & Cadastre','Marketing','Mechanical Engineering','Media and Communications','Micro-finance','Migration','Mining','Monitoring & Evaluation','Nuclear','Other','Pollution & Waste Management (incl. treatment)','Poverty Reduction','Printing','Procurement','Programme & Resource Management','Public Administration','Refrigeration','Regional Integration','Research','Risk Management (incl. insurance)','Roads & Bridges','Rural Development','Science & Innovation','Security','SME & Private Sector','Social Development','Standards & Consumer Protection','Statistics','Telecommunications','Tourism','Trade','Training','Translation','Transport','Urban Development','Vehicles','Water & Sanitation','Water Navigation & Ports & Shipping','Youth']
sector_code = [98,28,107,100,45,77,99,55,42,46,3,47,94,4,101,50,102,95,5,51,6,7,92,78,52,8,29,67,9,79,11,80,81,33,12,13,70,73,57,71,14,56,58,103,15,16,54,106,83,62,17,31,104,30,59,75,44,19,96,20,85,60,86,65,87,105,26,1,74,38,76,22,23,43,72,40,25,41,88,93,34,66,48,89,27]
sector_dict = {}
for i in range(0,len(sector_name)):
    sector_dict[sector_name[i]]=sector_code[i]

job_type_name = ["Contract, 12 months +", "Contract, 4 to 12 months","Contract, up to 4 months","Internship / Volunteer","Other","Permanent position"]
job_type_code = [3,2,1,5,6,4]
job_type_dict={}
for i in range(len(job_type_name)):
    job_type_dict[job_type_name[i]]=job_type_code[i]

organisation_type_name = ['Academic Institution','Consulting Organization','Engineering Firm','Financial Institution','Funding Agency','Government Agency','NGO','Other','Supplier']
organisation_type_code = [7,3,6,8,4,2,5,20,19,9]
organisation_type_dict = {}
for i in range(0,len(organisation_type_name)):
    organisation_type_dict[organisation_type_name[i]]=organisation_type_code[i]

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


def cast_net(name="",job_types="",languages="",countries="",minimum_experience="",maximum_experience="",recency="",sectors="",types=""):
    url = "https://www.developmentaid.org/jobs/search?sort=highlighted.desc,postedDate.desc"
    if name != '':
        name = name.replace(' ','%20')
        name = "&organizationName="+name
    if job_types != '':
        job_types_list = job_types.split(",")
        job_types_codes = []
        for job_type in job_types_list:
            if job_type in job_type_dict.keys():
                job_type = str(job_type_dict[job_type])
                job_types_codes.append(job_type)
            else:
                next
        addition = ""
        job_types = ''
        if job_types_codes != []:
            for code in job_types_codes:
                addition = addition + code + ','
            job_types = "&jobTypes="+addition[0:-1]
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
    if countries != '':
        country_list = country.split(",")
        country_codes = []
        for country in country_list:
            if country in country_dict.keys():
                country = str(country_dict[lang])
                country_codes.append(country)
            else:
                next
        addition = ""
        countries = ''
        if country_codes != []:
            for code in country_codes:
                addition = addition + code + ','
            countries = "&languages="+addition[0:-1]
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
        sector_list = sectors.split(",")
        sector_codes = []
        for sector in sector_list:
            if sector in sector_dict.keys():
                sect = str(sector_dict[sect])
                sector_codes.append(sect)
            else:
                next
        addition = ""
        sectors = ''
        if sector_codes != []:
            for code in sector_codes:
                addition = addition + code + ','
            sectors = "&sectors"+addition[0:-1]
    if types != '':
        organisation_type_list = types.split(",")
        organisation_type_codes = []
        for type in organisation_type_list:
            if organisation_type in organisation_type.keys():
                organisation_type = str(organisation_type_dict[organisation_type])
                organisation_type_codes.append(sect)
            else:
                next
        addition = ""
        organisation_types = ''
        if organisation_type_codes != []:
            for code in organisation_type_codes:
                addition = addition + code + ','
            organisation_types = "&types="+addition[0:-1]        
    home_page = url+name+job_types+languages+countries+minimum_experience+maximum_experience+recency+sectors+types
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
        op['Link to Tender'] = opportunity
        i+=1
    df = pd.DataFrame.from_dict(opportunity_dict, orient='index')
    excel_filename = 'Opportunities Summary'
    current_time = datetime.now()
    formatted_timestamp = current_time.strftime("%Y-%m-%d")
    new_string = excel_filename + " " + formatted_timestamp+'.xlsx'
    df.to_excel(new_string, index=False)
    print('Your potential jobs are now available at {}'.format(new_string))
def get_options(argument):
    options = ['name','job_types','languages','countries','minimum_experience','recency','sectors','types']
    if argument not in options: 
        return "Please select one of the cast_net criteria: names, job_types, languages, countries, minimum_experience, recency, sectors, or types"
    else:
        if argument == 'name':
            return "Please enter the name of the organisation whose job postings you are interested in seeing"
        elif argument == 'job_types':
            option_list = job_type_dict.keys()
            return "Types of jobs are: " + option_list
        elif argument == 'languages':
            option_list = language_dict.keys()
            return "Enter each language seperated only by a comma (no spaces). Language options are: "+option_list
        elif argument == 'countries':
            option_list = country_dict.keys()
            return "Valid country options are: " + option_list
        elif argument == 'minimum_experience':
            return "Please enter an integer (years) between 0 and 20"
        elif argument == 'recency':
            return "Please provide the maximum numbers of days since posting. Options are: 1,2,7,14,31"
        elif argument == 'sectors':
            option_list = region_dict.keys()
            return "Sector options are: " + option_list
        elif argument == 'types':
            option_list = organisation_type_dict.keys()
            return "Options for types of organisations are: " + option_list