import pandas as pd
from tqdm import tqdm
import json


def save_json_file(filename, dataframe, prompt, quick=False, task="classification"):
    objects = []
    for index, item in tqdm(enumerate(zip(dataframe['ID'].tolist(), dataframe['Image path'].tolist(), 
                                          dataframe['Description'].tolist(), dataframe["Command"].tolist(), 
                                          dataframe['Is command clear'].tolist(), dataframe['Expected output'].tolist())), 
                                          total=len(dataframe['ID'].tolist()), desc=f'Cycling {filename} data..'):
        game_id, image_path, description, command, is_command_clear, expected_output = item

        if task == "classification":
            value = is_command_clear
        elif task == "generation":
            value = expected_output
        elif task == "description":
            value = description
        else:
            raise ValueError(f"Task {task} not supported!")

        objects.append({
            "id": game_id,
            "image": image_path.split("/")[-1],
            "conversations": [
                {'from': 'human', 'value': prompt.replace("<command>", command)},
                {'from': 'gpt', 'value': value},
            ],
        })

        if quick and index == 5:
            break

    with open(filename, "w") as f:
        json.dump(objects, f)


# task could be 'classification' for the when to ask problem in IGLU
# or 'generation' for the what to ask in IGLU (i.e. to generate a question)
def create_iglu_json_instruct_data(task="classification", quick=False):
    df_train = pd.read_excel("./data/datasets/llava_train.xlsx")
    df_dev = pd.read_excel("./data/datasets/llava_dev.xlsx")
    df_test = pd.read_excel("./data/datasets/llava_test.xlsx")

    prompt_file_path = ""
    if task == "classification":
        prompt_file_path = "./data/instructions/iglu/when_to_ask_enriched"
    elif task == "generation":
        prompt_file_path = "./data/instructions/iglu/what_to_ask"
    elif task == "description":
        prompt_file_path = "./data/instructions/iglu/description"
    else:
        raise ValueError(f"Task {task} not supported!")
    
    prompt = ""
    with open(prompt_file_path, "r") as f:
        prompt = "".join(f.readlines())

    json_iglu_folder = "./data/datasets"
    save_json_file(f"{json_iglu_folder}/iglu_{task}_train.json", df_train, prompt, quick=quick, task=task)
    print(f"SAVED iglu_{task}_train.json")

    save_json_file(f"{json_iglu_folder}/iglu_{task}_dev.json", df_dev, prompt, quick=quick, task=task)
    print(f"SAVED iglu_{task}_dev.json")

    save_json_file(f"{json_iglu_folder}/iglu_{task}_test.json", df_test, prompt, quick=quick, task=task)
    print(f"SAVED iglu_{task}_test.json")

if __name__ == "__main__":
    create_iglu_json_instruct_data(quick=False, task="description")