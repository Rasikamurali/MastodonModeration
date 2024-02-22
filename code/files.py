import os
import json 
import pandas as pd 

# Directory containing the files
directory = r"C:\Users\rasik\Indiana University\Kamburugamuwa, Pasan - Mastodon Data\2023-10"


file_paths = [r"C:\Users\rasik\Indiana University\Kamburugamuwa, Pasan - Mastodon Data\2023-10\2023-10-11\mastodon.cloud_2023-10-11.json",
              r"C:\Users\rasik\Indiana University\Kamburugamuwa, Pasan - Mastodon Data\2023-10\2023-10-11\mastodon.social_2023-10-11.json",
              r"C:\Users\rasik\Indiana University\Kamburugamuwa, Pasan - Mastodon Data\2023-10\2023-10-12\genomic.social_2023-10-12.json",
            r"C:\Users\rasik\Indiana University\Kamburugamuwa, Pasan - Mastodon Data\2023-10\2023-10-12\mastodon.cloud_2023-10-12.json",
r"C:\Users\rasik\Indiana University\Kamburugamuwa, Pasan - Mastodon Data\2023-10\2023-10-12\mastodon.social_2023-10-12.json",
r"C:\Users\rasik\Indiana University\Kamburugamuwa, Pasan - Mastodon Data\2023-10\2023-10-12\mastodontech.de_2023-10-12.json",
r"C:\Users\rasik\Indiana University\Kamburugamuwa, Pasan - Mastodon Data\2023-10\2023-10-12\ruhr.social_2023-10-12.json",
r"C:\Users\rasik\Indiana University\Kamburugamuwa, Pasan - Mastodon Data\2023-10\2023-10-12\sfba.social_2023-10-12.json",
r"C:\Users\rasik\Indiana University\Kamburugamuwa, Pasan - Mastodon Data\2023-10\2023-10-13\genomic.social_2023-10-13.json",
r"C:\Users\rasik\Indiana University\Kamburugamuwa, Pasan - Mastodon Data\2023-10\2023-10-13\mastodon.cloud_2023-10-13.json",
r"C:\Users\rasik\Indiana University\Kamburugamuwa, Pasan - Mastodon Data\2023-10\2023-10-13\mastodon.social_2023-10-13.json",
r"C:\Users\rasik\Indiana University\Kamburugamuwa, Pasan - Mastodon Data\2023-10\2023-10-13\mastodontech.de_2023-10-13.json",
r"C:\Users\rasik\Indiana University\Kamburugamuwa, Pasan - Mastodon Data\2023-10\2023-10-13\ruhr.social_2023-10-13.json",
r"C:\Users\rasik\Indiana University\Kamburugamuwa, Pasan - Mastodon Data\2023-10\2023-10-13\sfba.social_2023-10-13.json",
r"C:\Users\rasik\Indiana University\Kamburugamuwa, Pasan - Mastodon Data\2023-10\2023-10-14\genomic.social_2023-10-14.json",
r"C:\Users\rasik\Indiana University\Kamburugamuwa, Pasan - Mastodon Data\2023-10\2023-10-14\mastodon.cloud_2023-10-14.json",
r"C:\Users\rasik\Indiana University\Kamburugamuwa, Pasan - Mastodon Data\2023-10\2023-10-14\mastodon.social_2023-10-14.json",
r"C:\Users\rasik\Indiana University\Kamburugamuwa, Pasan - Mastodon Data\2023-10\2023-10-14\mastodontech.de_2023-10-14.json",
r"C:\Users\rasik\Indiana University\Kamburugamuwa, Pasan - Mastodon Data\2023-10\2023-10-14\ruhr.social_2023-10-14.json",
r"C:\Users\rasik\Indiana University\Kamburugamuwa, Pasan - Mastodon Data\2023-10\2023-10-14\sfba.social_2023-10-14.json",
r"C:\Users\rasik\Indiana University\Kamburugamuwa, Pasan - Mastodon Data\2023-10\2023-10-15\genomic.social_2023-10-15.json",
r"C:\Users\rasik\Indiana University\Kamburugamuwa, Pasan - Mastodon Data\2023-10\2023-10-15\mastodon.cloud_2023-10-15.json",
r"C:\Users\rasik\Indiana University\Kamburugamuwa, Pasan - Mastodon Data\2023-10\2023-10-15\mastodon.social_2023-10-15.json",
r"C:\Users\rasik\Indiana University\Kamburugamuwa, Pasan - Mastodon Data\2023-10\2023-10-15\mastodontech.de_2023-10-15.json",
r"C:\Users\rasik\Indiana University\Kamburugamuwa, Pasan - Mastodon Data\2023-10\2023-10-15\ruhr.social_2023-10-15.json",
r"C:\Users\rasik\Indiana University\Kamburugamuwa, Pasan - Mastodon Data\2023-10\2023-10-15\sfba.social_2023-10-15.json",
r"C:\Users\rasik\Indiana University\Kamburugamuwa, Pasan - Mastodon Data\2023-10\2023-10-16\genomic.social_2023-10-16.json",
r"C:\Users\rasik\Indiana University\Kamburugamuwa, Pasan - Mastodon Data\2023-10\2023-10-16\mastodon.cloud_2023-10-16.json",
r"C:\Users\rasik\Indiana University\Kamburugamuwa, Pasan - Mastodon Data\2023-10\2023-10-16\mastodon.social_2023-10-16.json",
r"C:\Users\rasik\Indiana University\Kamburugamuwa, Pasan - Mastodon Data\2023-10\2023-10-16\mastodontech.de_2023-10-16.json",
r"C:\Users\rasik\Indiana University\Kamburugamuwa, Pasan - Mastodon Data\2023-10\2023-10-16\ruhr.social_2023-10-16.json",
r"C:\Users\rasik\Indiana University\Kamburugamuwa, Pasan - Mastodon Data\2023-10\2023-10-16\sfba.social_2023-10-16.json",
r"C:\Users\rasik\Indiana University\Kamburugamuwa, Pasan - Mastodon Data\2023-10\2023-10-17\genomic.social_2023-10-17.json",
r"C:\Users\rasik\Indiana University\Kamburugamuwa, Pasan - Mastodon Data\2023-10\2023-10-17\mastodon.cloud_2023-10-17.json",
r"C:\Users\rasik\Indiana University\Kamburugamuwa, Pasan - Mastodon Data\2023-10\2023-10-17\mastodon.social_2023-10-17.json",
r"C:\Users\rasik\Indiana University\Kamburugamuwa, Pasan - Mastodon Data\2023-10\2023-10-17\mastodontech.de_2023-10-17.json",
r"C:\Users\rasik\Indiana University\Kamburugamuwa, Pasan - Mastodon Data\2023-10\2023-10-17\ruhr.social_2023-10-17.json",
r"C:\Users\rasik\Indiana University\Kamburugamuwa, Pasan - Mastodon Data\2023-10\2023-10-17\sfba.social_2023-10-17.json",
r"C:\Users\rasik\Indiana University\Kamburugamuwa, Pasan - Mastodon Data\2023-10\2023-10-18\genomic.social_2023-10-18.json",
r"C:\Users\rasik\Indiana University\Kamburugamuwa, Pasan - Mastodon Data\2023-10\2023-10-18\mastodon.cloud_2023-10-18.json",
r"C:\Users\rasik\Indiana University\Kamburugamuwa, Pasan - Mastodon Data\2023-10\2023-10-18\mastodon.social_2023-10-18.json",
r"C:\Users\rasik\Indiana University\Kamburugamuwa, Pasan - Mastodon Data\2023-10\2023-10-18\mastodontech.de_2023-10-18.json",
r"C:\Users\rasik\Indiana University\Kamburugamuwa, Pasan - Mastodon Data\2023-10\2023-10-18\ruhr.social_2023-10-18.json",
r"C:\Users\rasik\Indiana University\Kamburugamuwa, Pasan - Mastodon Data\2023-10\2023-10-18\sfba.social_2023-10-18.json",
r"C:\Users\rasik\Indiana University\Kamburugamuwa, Pasan - Mastodon Data\2023-10\2023-10-19\genomic.social_2023-10-19.json",
r"C:\Users\rasik\Indiana University\Kamburugamuwa, Pasan - Mastodon Data\2023-10\2023-10-19\mastodon.cloud_2023-10-19.json",
r"C:\Users\rasik\Indiana University\Kamburugamuwa, Pasan - Mastodon Data\2023-10\2023-10-19\mastodon.social_2023-10-19.json",
r"C:\Users\rasik\Indiana University\Kamburugamuwa, Pasan - Mastodon Data\2023-10\2023-10-19\mastodontech.de_2023-10-19.json",
r"C:\Users\rasik\Indiana University\Kamburugamuwa, Pasan - Mastodon Data\2023-10\2023-10-19\ruhr.social_2023-10-19.json",
r"C:\Users\rasik\Indiana University\Kamburugamuwa, Pasan - Mastodon Data\2023-10\2023-10-19\sfba.social_2023-10-19.json",
r"C:\Users\rasik\Indiana University\Kamburugamuwa, Pasan - Mastodon Data\2023-10\2023-10-20\genomic.social_2023-10-20.json",
r"C:\Users\rasik\Indiana University\Kamburugamuwa, Pasan - Mastodon Data\2023-10\2023-10-20\mastodon.cloud_2023-10-20.json",
r"C:\Users\rasik\Indiana University\Kamburugamuwa, Pasan - Mastodon Data\2023-10\2023-10-20\mastodon.social_2023-10-20.json",
r"C:\Users\rasik\Indiana University\Kamburugamuwa, Pasan - Mastodon Data\2023-10\2023-10-20\mastodontech.de_2023-10-20.json",
r"C:\Users\rasik\Indiana University\Kamburugamuwa, Pasan - Mastodon Data\2023-10\2023-10-20\ruhr.social_2023-10-20.json",
r"C:\Users\rasik\Indiana University\Kamburugamuwa, Pasan - Mastodon Data\2023-10\2023-10-20\sfba.social_2023-10-20.json",
r"C:\Users\rasik\Indiana University\Kamburugamuwa, Pasan - Mastodon Data\2023-10\2023-10-21\genomic.social_2023-10-21.json",
r"C:\Users\rasik\Indiana University\Kamburugamuwa, Pasan - Mastodon Data\2023-10\2023-10-21\mastodon.cloud_2023-10-21.json",
r"C:\Users\rasik\Indiana University\Kamburugamuwa, Pasan - Mastodon Data\2023-10\2023-10-21\mastodon.social_2023-10-21.json",
r"C:\Users\rasik\Indiana University\Kamburugamuwa, Pasan - Mastodon Data\2023-10\2023-10-21\mastodontech.de_2023-10-21.json",
r"C:\Users\rasik\Indiana University\Kamburugamuwa, Pasan - Mastodon Data\2023-10\2023-10-21\ruhr.social_2023-10-21.json",
r"C:\Users\rasik\Indiana University\Kamburugamuwa, Pasan - Mastodon Data\2023-10\2023-10-21\sfba.social_2023-10-21.json",
r"C:\Users\rasik\Indiana University\Kamburugamuwa, Pasan - Mastodon Data\2023-10\2023-10-22\genomic.social_2023-10-22.json",
r"C:\Users\rasik\Indiana University\Kamburugamuwa, Pasan - Mastodon Data\2023-10\2023-10-22\mastodon.cloud_2023-10-22.json",
r"C:\Users\rasik\Indiana University\Kamburugamuwa, Pasan - Mastodon Data\2023-10\2023-10-22\mastodon.social_2023-10-22.json",
r"C:\Users\rasik\Indiana University\Kamburugamuwa, Pasan - Mastodon Data\2023-10\2023-10-22\mastodontech.de_2023-10-22.json",
r"C:\Users\rasik\Indiana University\Kamburugamuwa, Pasan - Mastodon Data\2023-10\2023-10-22\ruhr.social_2023-10-22.json",
r"C:\Users\rasik\Indiana University\Kamburugamuwa, Pasan - Mastodon Data\2023-10\2023-10-22\sfba.social_2023-10-22.json",
r"C:\Users\rasik\Indiana University\Kamburugamuwa, Pasan - Mastodon Data\2023-10\2023-10-23\genomic.social_2023-10-23.json",
r"C:\Users\rasik\Indiana University\Kamburugamuwa, Pasan - Mastodon Data\2023-10\2023-10-23\mastodon.cloud_2023-10-23.json",
r"C:\Users\rasik\Indiana University\Kamburugamuwa, Pasan - Mastodon Data\2023-10\2023-10-23\mastodon.social_2023-10-23.json",
r"C:\Users\rasik\Indiana University\Kamburugamuwa, Pasan - Mastodon Data\2023-10\2023-10-23\mastodontech.de_2023-10-23.json",
r"C:\Users\rasik\Indiana University\Kamburugamuwa, Pasan - Mastodon Data\2023-10\2023-10-23\ruhr.social_2023-10-23.json",
r"C:\Users\rasik\Indiana University\Kamburugamuwa, Pasan - Mastodon Data\2023-10\2023-10-23\sfba.social_2023-10-23.json",
r"C:\Users\rasik\Indiana University\Kamburugamuwa, Pasan - Mastodon Data\2023-10\2023-10-24\genomic.social_2023-10-24.json",
r"C:\Users\rasik\Indiana University\Kamburugamuwa, Pasan - Mastodon Data\2023-10\2023-10-24\mastodon.cloud_2023-10-24.json",
r"C:\Users\rasik\Indiana University\Kamburugamuwa, Pasan - Mastodon Data\2023-10\2023-10-24\mastodon.social_2023-10-24.json",
r"C:\Users\rasik\Indiana University\Kamburugamuwa, Pasan - Mastodon Data\2023-10\2023-10-24\mastodontech.de_2023-10-24.json",
r"C:\Users\rasik\Indiana University\Kamburugamuwa, Pasan - Mastodon Data\2023-10\2023-10-24\ruhr.social_2023-10-24.json",
r"C:\Users\rasik\Indiana University\Kamburugamuwa, Pasan - Mastodon Data\2023-10\2023-10-24\sfba.social_2023-10-24.json",
r"C:\Users\rasik\Indiana University\Kamburugamuwa, Pasan - Mastodon Data\2023-10\2023-10-25\genomic.social_2023-10-25.json",
r"C:\Users\rasik\Indiana University\Kamburugamuwa, Pasan - Mastodon Data\2023-10\2023-10-25\mastodon.cloud_2023-10-25.json",
r"C:\Users\rasik\Indiana University\Kamburugamuwa, Pasan - Mastodon Data\2023-10\2023-10-25\mastodon.social_2023-10-25.json",
r"C:\Users\rasik\Indiana University\Kamburugamuwa, Pasan - Mastodon Data\2023-10\2023-10-25\mastodontech.de_2023-10-25.json",
r"C:\Users\rasik\Indiana University\Kamburugamuwa, Pasan - Mastodon Data\2023-10\2023-10-25\ruhr.social_2023-10-25.json",
r"C:\Users\rasik\Indiana University\Kamburugamuwa, Pasan - Mastodon Data\2023-10\2023-10-25\sfba.social_2023-10-25.json",
r"C:\Users\rasik\Indiana University\Kamburugamuwa, Pasan - Mastodon Data\2023-10\2023-10-26\genomic.social_2023-10-26.json",
r"C:\Users\rasik\Indiana University\Kamburugamuwa, Pasan - Mastodon Data\2023-10\2023-10-26\mastodon.cloud_2023-10-26.json",
r"C:\Users\rasik\Indiana University\Kamburugamuwa, Pasan - Mastodon Data\2023-10\2023-10-26\mastodon.social_2023-10-26.json",
r"C:\Users\rasik\Indiana University\Kamburugamuwa, Pasan - Mastodon Data\2023-10\2023-10-26\mastodontech.de_2023-10-26.json",
r"C:\Users\rasik\Indiana University\Kamburugamuwa, Pasan - Mastodon Data\2023-10\2023-10-26\ruhr.social_2023-10-26.json",
r"C:\Users\rasik\Indiana University\Kamburugamuwa, Pasan - Mastodon Data\2023-10\2023-10-26\sfba.social_2023-10-26.json"
]

 
# for file in file_paths: 
#     data = []
#     with open(file, 'r') as f: 
#         for line in f: 
#             post = json.loads(line)
#             data.append(post)
#     df = pd.DataFrame(data)
#     print(df.head())  

for file_path in file_paths: 
    print(list(file_path.split("\\"))[-1])

for file_path in file_paths:
    name = list(file_path.split("\\"))[-1]
    data = []
    try:
        with open(file_path, 'r') as file:
            for line in file:
                post = json.loads(line)
                data.append(post)
        df = pd.DataFrame(data)
        df.to_csv(f"{name}.csv")
        print(f"DataFrame for {file_path}:")
        print(df.head())
    except FileNotFoundError:
        print(f"File '{file_path}' not found.")
    except Exception as e:
        print(f"An error occurred while processing '{file_path}': {str(e)}")