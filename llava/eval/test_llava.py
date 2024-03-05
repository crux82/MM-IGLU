import argparse
import torch
import pandas as pd

from PIL import Image
from tqdm import tqdm

from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN
from llava.model.builder import load_pretrained_model
from llava.utils import disable_torch_init
from llava.mm_utils import tokenizer_image_token, get_model_name_from_path, KeywordsStoppingCriteria


def load_image(image_file):
    if image_file.startswith('http') or image_file.startswith('https'):
        response = requests.get(image_file)
        image = Image.open(BytesIO(response.content)).convert('RGB')
    else:
        image = Image.open(image_file).convert('RGB')
    return image


def compute_confusion_matrix(expected_outputs, outputs_list, task="classification"):

    positive_class = "I can execute it."

    positive_class_cm = {
        "tp": 0,
        "fn": 0,
        "fp": 0,
        "precision": 0,
        "recall": 0,
        "F1": 0
    }
    negative_class_cm = {
        "tp": 0,
        "fn": 0,
        "fp": 0,
        "precision": 0,
        "recall": 0,
        "F1": 0
    }
    for pred, truth in zip(outputs_list, expected_outputs):
        if truth == positive_class:
            if pred == truth:
                positive_class_cm['tp'] += 1
            else:
                positive_class_cm['fn'] += 1
                negative_class_cm['fp'] += 1
        
        elif pred == positive_class:
            # truth != pred
            positive_class_cm['fp'] += 1

            # truth == negative_class
            negative_class_cm['fn'] += 1
        
        else:
            negative_class_cm['tp'] += 1

    positive_class_cm['precision'] = positive_class_cm['tp'] / (positive_class_cm['tp'] + positive_class_cm['fp'])
    positive_class_cm['recall'] = positive_class_cm['tp'] / (positive_class_cm['tp'] + positive_class_cm['fn'])
    positive_class_cm['F1'] = (2 * positive_class_cm['precision'] * positive_class_cm['recall']) / (positive_class_cm['precision'] + positive_class_cm['recall'])
    
    negative_class_cm['precision'] = negative_class_cm['tp'] / (negative_class_cm['tp'] + negative_class_cm['fp'])
    negative_class_cm['recall'] = negative_class_cm['tp'] / (negative_class_cm['tp'] + negative_class_cm['fn'])
    negative_class_cm['F1'] = (2 * negative_class_cm['precision'] * negative_class_cm['recall']) / (negative_class_cm['precision'] + negative_class_cm['recall'])

    macro_f1 = (positive_class_cm['F1'] + negative_class_cm['F1']) / 2

    df = pd.DataFrame({
        "Positive Class Header": ["TP", "FP", "FN", "", "Precision", "Recall", "F1", "", "", "Macro F1"],
        "Positive Class Value": [positive_class_cm['tp'], positive_class_cm['fp'], positive_class_cm['fn'], "", positive_class_cm['precision'], positive_class_cm['recall'], positive_class_cm['F1'], "", "", macro_f1],
        "Negative Class Header": ["TP", "FP", "FN", "", "Precision", "Recall", "F1", "", "", ""],
        "Negative Class Value": [negative_class_cm['tp'], negative_class_cm['fp'], negative_class_cm['fn'], "", negative_class_cm['precision'], negative_class_cm['recall'], negative_class_cm['F1'], "", "", ""],
    })

    return df




def test_model(args):
    # Model
    disable_torch_init()

    model_name = get_model_name_from_path(args.model_path)
    tokenizer, model, image_processor, context_len = load_pretrained_model(args.model_path, args.model_base, model_name)

    prompt_file_path = ""
    if args.task == "classification":
        prompt_file_path = "./data/instructions/iglu/when_to_ask_enriched"
    elif args.task == "generation":
        prompt_file_path = "./data/instructions/iglu/what_to_ask"
    elif args.task == "description":
        prompt_file_path = "./data/instructions/iglu/description"
    else:
        raise ValueError(f"TASK {args.task} not supported!")
    
    prompt_base = ""
    with open(prompt_file_path, "r") as f:
        prompt_base = "".join(f.readlines())

    input_file_path = args.input_file_path
    dataframe = pd.read_excel(input_file_path)
    expected_outputs, outputs_list, perfect_matches, image_file_paths_list = [], [], [], []
    for index, item in tqdm(enumerate(zip(dataframe['ID'].tolist(), dataframe['Description'].tolist(),
                                          dataframe["Command"].tolist(), dataframe['Is command clear'].tolist(), 
                                          dataframe['Expected output'].tolist())), total=len(dataframe['ID'].tolist()), 
                                          desc=f'Cycling {args.input_file_path} data..'):
        game_id, description, command, is_command_clear, answer = item
        prompt_i = prompt_base.replace("<command>", command).replace("<image>", DEFAULT_IMAGE_TOKEN)

        image_file_path = f"{args.image_path}/{game_id}.png"
        image_file_paths_list.append(image_file_path)
        image = load_image(image_file_path)
        image_tensor = image_processor.preprocess(image, return_tensors='pt')['pixel_values'].half().cuda()
        input_ids = tokenizer_image_token(prompt_i, tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt').unsqueeze(0).cuda()
        
        stop_str = "</s>"
        stopping_criteria = KeywordsStoppingCriteria([stop_str], tokenizer, input_ids)

        with torch.inference_mode():
            output_ids = model.generate(
                input_ids,
                images=image_tensor,
                do_sample=True,
                temperature=0.2,
                max_new_tokens=1024,
                use_cache=True,
                stopping_criteria=[stopping_criteria])

        input_token_len = input_ids.shape[1]
        n_diff_input_output = (input_ids != output_ids[:, :input_token_len]).sum().item()
        if n_diff_input_output > 0:
            print(f'[Warning] {n_diff_input_output} output_ids are not the same as the input_ids')
        outputs = tokenizer.batch_decode(output_ids[:, input_token_len:], skip_special_tokens=True)[0]
        outputs = outputs.strip()
        if outputs.endswith(stop_str):
            outputs = outputs[:-len(stop_str)]
        outputs = outputs.strip()

        if args.task == "classification":
            outputs = "I can execute it." if outputs == "Yes" else outputs
            perfect_match = 1 if outputs == answer else 0
            expected_output = answer
        elif args.task == "generation":
            expected_output = answer
            perfect_match = 1 if outputs == answer else 0
        elif args.task == "description":
            expected_output = description
            perfect_match = 1 if outputs == description else 0
        
        expected_outputs.append(expected_output)
        outputs_list.append(outputs)
        perfect_matches.append(perfect_match)

        if args.quick_test:
            print(50*"*")
            print(f"PROMPT: {prompt_i}")
            print(outputs)
            print()
            print(f"Expected output: {expected_output}")
            print(50*"*")
            if index == 10:
                quit()

    dataframe['Model output'] = outputs_list
    dataframe['Image Path'] = image_file_paths_list
    dataframe['Perfect match'] = perfect_matches

    if args.task == "classification" or args.task == "generation":
        cm_df = compute_confusion_matrix(expected_outputs, outputs_list, task=args.task)

    file_to_save = f"./data/predictions/{args.model_name}_{args.task}.xlsx"
    
    # create a excel writer object
    with pd.ExcelWriter(file_to_save, engine="openpyxl") as writer:
        dataframe.to_excel(writer, sheet_name="Generations", index=False)
        
        if args.task == "classification" or args.task == "generation":
            cm_df.to_excel(writer, sheet_name="Performance", index=False)

    print(f"SAVED {file_to_save}")



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, required=True)
    parser.add_argument("--model-base", type=str, required=True)
    parser.add_argument("--model-name", type=str, required=True)
    parser.add_argument("--input-file-path", type=str, required=True)
    parser.add_argument("--image-path", type=str, required=True)
    parser.add_argument("--task", type=str, required=True)
    parser.add_argument("--quick-test", type=bool, default=False)
    args = parser.parse_args()

    test_model(args)
