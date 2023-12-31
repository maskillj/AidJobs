# AidJobs
## About
AidJobs is a AI-powered tool for summarising relevant job opportunities in the humanitarian and development sector. Users receive an Excel file with an AI summary of the key details of relevant project opportunities from sources where information is not uniformly presented.

## How Does It Work?
The central function of the AidJobs Python file, 'cast_net', searches for all jobs meeting the user's criteria as provided as arguments. The pre-trained Q&A language model, roberta-base-squad2, is then used to find relevant information for each information type. The model is available at: https://huggingface.co/deepset/roberta-base-squad2.

## Instructions for Use

After downloading the relevant packages as specified in 'requirements.txt', the function 'cast_net()' can be used to search through available jobs to find appropriate opportunities. This function comes with a number of optional arguments:
#### name - The name of a listing organisation (e.g., 'Save the Children')
#### job_types – The type of the opportunity sought. Any combination of the following: Contract, 12 months +; Contract, 4 to 12 months; Contract, up to 4 months; Internship / Volunteer; Permanent position; or Other
#### languages - Any language requirements for which you wish to see open positions (e.g. 'Spanish')
#### countries - Any countries of interest for which you wish to see open positions (e.g. 'Cameroon')
#### minimum experience - Minimum experience required for open positions
#### sectors - Sectors in which you wish to see open positions (e.g., 'Education')
#### types - The type of agency listing open positions (e.g., 'Funding Agency')
For a full list of options available for each argument, please run 'get_options()' with the argument name as the argument. 
