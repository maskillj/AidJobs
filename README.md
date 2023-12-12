# AidJobsAssets
## About
AidJobs is a AI-powered tool for summarising relevant job opportunities in the humanitarian and development sector. Users receive an Excel file with an AI summary of the key details of relevant project opportunities from sources where information is not uniformly presented.

## How Does It Work?
The central function of the AidJobs Python file, 'cast_net', searches for all jobs meeting the user's criteria as provided as arguments. The pre-trained Q&A language model, roberta-base-squad2, is then used to find relevant information for each information type. The model is available at: https://huggingface.co/deepset/roberta-base-squad2.

## Instructions for Use
