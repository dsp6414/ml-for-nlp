import os
import pandas as pd
import requests
import zipfile
import json
import io
import pdb

def output_targets(model_name, exp_name):
    exp_dir = os.path.join('experiments', exp_name)
    exp_file_name = 'abstract.results.base_' + model_name +'.txt'
    exp_file_path = os.path.join(exp_dir, exp_file_name)
    targets = pd.read_csv(exp_file_path)
    return targets['target'].values, targets['similarity'].values


def get_target(targets, similarity, question_num):
    return targets[question_num], similarity[question_num]

def score_qualtrics_csv(model_name, exp_name):
    csv_dir = os.path.join('qualtrics', exp_name)
    csv_path = os.path.join(csv_dir, model_name + '.csv')
    db = pd.read_csv(csv_path)
    # The submissions are in the last 100 rows.
    submissions = db.iloc[2:, 12:112] # lol

    n_correct = 0.0
    n_total = 0.0

    targets, similarity = output_targets(model_name, exp_name)

    by_sim_correct = {1: 0, 2: 0, 3: 0, 4: 0}
    by_sim_total = {1: 0, 2: 0, 3: 0, 4: 0}
    for question_num in range(100):
        # Get that column
        question_subs = submissions[str(question_num + 1)] # Qualtrics is 1-100 rather than 0-99
        # Correct target
        correct_target, correct_sim = get_target(targets, similarity, question_num)
        for x in question_subs.values:
            if pd.isnull(x):
                pass
                # don't do anything
            elif correct_target in x:
                # Correct, thank god
                by_sim_correct[correct_sim] += 1
                by_sim_total[correct_sim] += 1
                n_correct += 1
                n_total += 1
            else:
                # Rip
                by_sim_total[correct_sim] += 1
                n_total += 1

    accuracy = n_correct / n_total
    print('Model Name: %s, Experiment Name: %s, Accuracy: %f' % (model_name, exp_name, accuracy))
    for num, total in by_sim_total.items():
        print(str(num) + ' Similar: %f' % (by_sim_correct[num] / by_sim_total[num]))

# print(score_qualtrics_csv('ss1252', 'by_similarity'))


def qualtrics_name_from_num(model_num):
    qualtrics_name = 'CS287: Pragmatic Image Captioning (%s).csv' % (model_num)
    return qualtrics_name

def model_num_from_name(model_name):
    if model_name.startswith('ss1'):
        return model_name[3:]
    elif model_name.startswith('s0'):
        return model_name[2:]
    else:
        return None

def download_file(surveyId, model_name, exp_name):
    # model name: 'ss1252', etc.
    # exp_name: 'by_similarity' or whatever
    # Setting user Parameters
    apiToken = "vCC3UDC8iFk2BWDZsx05xNJOtX1Rty6q0azLSlBT"

    fileFormat = "csv"
    dataCenter = 'az1'

    # Setting static parameters
    requestCheckProgress = 0
    progressStatus = "in progress"
    baseUrl = "https://{0}.qualtrics.com/API/v3/responseexports/".format(dataCenter)
    headers = {
        "content-type": "application/json",
        "x-api-token": apiToken,
        }

    # Step 1: Creating Data Export
    downloadRequestUrl = baseUrl
    downloadRequestPayload = ('{"format":"%s",'
        '"surveyId":"%s",'
        '"useLabels":true}' % (fileFormat, surveyId)
        )
    old_downloadRequestPayload = '{"format":"' + fileFormat + '","surveyId":"' + surveyId + '"}'

    downloadRequestResponse = requests.request("POST", downloadRequestUrl, data=downloadRequestPayload, headers=headers)
    progressId = downloadRequestResponse.json()["result"]["id"]
    # print(downloadRequestResponse.text)

    # Step 2: Checking on Data Export Progress and waiting until export is ready
    while requestCheckProgress < 100 and progressStatus is not "complete":
        requestCheckUrl = baseUrl + progressId
        requestCheckResponse = requests.request("GET", requestCheckUrl, headers=headers)
        requestCheckProgress = requestCheckResponse.json()["result"]["percentComplete"]
        # print("Download is " + str(requestCheckProgress) + " complete")

    # Step 3: Downloading file
    requestDownloadUrl = baseUrl + progressId + '/file'
    requestDownload = requests.request("GET", requestDownloadUrl, headers=headers, stream=True)

    # Step 4: Unzipping the file
    zipfile.ZipFile(io.BytesIO(requestDownload.content)).extractall("MyQualtricsDownload")
    # print('Complete')

    # Step 5: Rename and move
    model_num = model_num_from_name(model_name)
    extracted_path = os.path.join('MyQualtricsDownload', qualtrics_name_from_num(model_num))
    new_path = os.path.join('qualtrics', exp_name, model_name +'.csv')
    os.rename(extracted_path, new_path)

# surveyId = "SV_9v14qCgBotNEvPv" # 251
# "SV_e985tYRmNHdZGG9" # 252

# surveys = [("SV_e985tYRmNHdZGG9", 'ss1252', 'by_similarity'), 
#             ("SV_9v14qCgBotNEvPv", 'ss1251', 'one_different'),
#             ("SV_9Y7wryBq8bVE5nf", 's0279', 'by_similarity'),
#             ("SV_bBJecbwsZKFaKtD", 's0280', 'one_different')]

surveys = [("SV_e985tYRmNHdZGG9", 'ss1252', 'by_similarity'), 
            ("SV_9Y7wryBq8bVE5nf", 's0279', 'by_similarity')]

for surveyId, model_name, exp_name in surveys:
    download_file(surveyId, model_name, exp_name)
    score_qualtrics_csv(model_name, exp_name)

#download_file(surveyId, 'ss1251', 'one_different')
#print(score_qualtrics_csv('ss1251', 'one_different'))
