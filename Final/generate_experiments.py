import os
import random
import pandas as pd
import pdb

random.seed(1)

def gen_dropbox_link(image_id):
    dropbox_link = 'https://s3.amazonaws.com/cs287project/by_similarity/Scene' + image_id + '.png'
    return dropbox_link 

def experiment_to_qualtrics(model_name, exp_name): 
    # exp_name: 'by_similarity', 'one_different'
    exp_dir = os.path.join('experiments', exp_name)
    exp_file_name = 'abstract.results.base_' + model_name +'.txt'
    exp_file_path = os.path.join(exp_dir, exp_file_name)
    out_dir = os.path.join('qualtrics', exp_name)
    qualtrics_file_path = os.path.join(out_dir, model_name + '.txt')
    with open(exp_file_path) as infile, open(qualtrics_file_path, 'w') as out_file:
        next(infile) # skip first line
        for line in infile:
            line_split = line.split(',')
            row,target,distractor,similarity,model_name,score,description = line_split
            # Switch the order of images randomly
            images = [target, distractor]
            random.shuffle(images)
            # Write description as question. 
            out_file.write("%d. Desciption : %s" % (int(row) + 1, description)) 
            out_file.write("\n")
            out_file.write("Image choice 1: <a href=\"%s\" target=\"_blank\">%s</a> \n" % (
                gen_dropbox_link(images[0]), images[0]))
            out_file.write("Image choice 2: <a href=\"%s\" target=\"_blank\">%s</a> \n" % (
                gen_dropbox_link(images[1]), images[1]))
            out_file.write("\n")

# experiment_to_qualtrics('ss1252', 'by_similarity')
# experiment_to_qualtrics('ss1251', 'one_different')
# experiment_to_qualtrics('s0280', 'one_different')
# experiment_to_qualtrics('s0279', 'by_similarity')


# print(score_qualtrics_csv('ss1252', 'by_similarity'))