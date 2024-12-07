import torch


data_path = "/home/zelin3/Data/CelebA-HQ"
fake_path = "/home/zelin3/projects/Modified-Anti-DreamBooth/test_authorized_freq"
import pickle


import glob
prompt = 'a_photo_of_sks_person'
avg_id_set = {}
for item in ['checkpoint-1000']:
    result = {
    "a_photo_of_sks_person": {
            "ism": [],
            "fdfr": []
        }
    }
    for idx in os.listdir(data_path):
        idx_data_dir = os.path.join(data_path, idx, "set_A"), os.path.join(data_path, idx, "set_B")
        mid_dir = "{}/DREAMBOOTH/{}/dreambooth".format(idx,item)
        idx_fake_dir = os.path.join(fake_path, mid_dir, prompt)
        ism, fdfr,avg_embedding = matching_score_genimage_id(idx_fake_dir, idx_data_dir)
        avg_id_set[idx] = avg_embedding
        result[prompt]["fdfr"].append(fdfr)
        if ism is None:
            continue
        result[prompt]["ism"].append(ism)
        

    print("{} ism: {}".format(item, torch.mean(torch.stack(result[prompt]["ism"]))))
    print("{} fdfr: {}".format(item, torch.mean(torch.tensor(result[prompt]["fdfr"]))))
with open('/home/zelin3/Data/celeba_avg_id','wb') as f:
    pickle.dump(avg_id_set,f)