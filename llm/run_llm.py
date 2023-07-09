"""Send a test message."""
import argparse
import json
import tqdm

import requests

from fastchat.model.model_adapter import get_conversation_template


def main():
    model_name = args.model_name

    if args.worker_address:
        worker_addr = args.worker_address
    else:
        controller_addr = args.controller_address
        ret = requests.post(controller_addr + "/refresh_all_workers")
        ret = requests.post(controller_addr + "/list_models")
        models = ret.json()["models"]
        models.sort()
        print(f"Models: {models}")

        ret = requests.post(
            controller_addr + "/get_worker_address", json={"model": model_name}
        )
        worker_addr = ret.json()["address"]
        print(f"worker_addr: {worker_addr}")

    if worker_addr == "":
        print(f"No available workers for {model_name}")
        return
    
    results = []
    test_data = json.load(open('./dacon_submissions/test_qa_albert-xxlarge-v2.json', 'r', encoding='utf-8'))

    for data in tqdm.tqdm(test_data):
        message = """
        Predict the answer of the last question based on the given Q&A examples. You have to strictly follow the answer format.

        """

        qa_index = 1
        qa_template = """
        Q{0}. Who is the winner among "{1}" and "{2}" based on the fact? The fact: {3}
        A{0}. {4}
        """

        for qa_data in data['victory_examples'] + data['defeat_examples']:
            first = qa_data['The first party']
            second = qa_data['The second party']

            message += qa_template.format(
                qa_index, 
                qa_data['The first party'],
                qa_data['The second party'],
                qa_data['facts'].replace('\n', ' '),
                f'The winner is "{first}"' if qa_data['gt'] == 'Victory' else f'The winner is "{second}"'
            ); qa_index += 1

        # message += f"""
        # Q{qa_index}. Based on the fact, who is the winner among "{data['The first party']}" and "{data['The second party']}"?
        # * The fact: {data['facts']}
        # A{qa_index}. 
        # """

        fact = data['facts'].replace('\n', ' ')
        message += f"""
        Q{qa_index}. Who is the winner among "{data['The first party']}" and "{data['The second party']}" based on the fact? The fact: {fact}
        A{qa_index}. """

        conv = get_conversation_template(model_name)
        conv.append_message(conv.roles[0], message)
        conv.append_message(conv.roles[1], None)
        prompt = conv.get_prompt()

        headers = {"User-Agent": "FastChat Client"}
        gen_params = {
            "model": model_name,
            "prompt": prompt,
            "temperature": args.temperature,
            "max_new_tokens": args.max_new_tokens,
            "stop": conv.stop_str,
            "stop_token_ids": conv.stop_token_ids,
            "echo": False,
        }
        response = requests.post(
            worker_addr + "/worker_generate",
            headers=headers,
            json=gen_params,
            stream=False,
        )
        print(response)
        response = response.json()

        # input()
        
        del data['victory_examples']
        del data['defeat_examples']

        prediction = str(response['text'])
        # if '\n' in prediction:
        #     prediction = prediction.split('\n')[-1]

        data['prediction'] = prediction
        results.append(data)

        print(f"{conv.roles[0]}: {message}")
        print(f"{conv.roles[1]}: {prediction}")

        json.dump(results, open(f'./dacon_submissions/test_qa_albert-xxlarge-v2_{model_name}.json', 'w', encoding='utf-8'), indent='\t', ensure_ascii=False)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--controller-address", type=str, default="http://localhost:21001"
    )
    parser.add_argument("--worker-address", type=str)
    parser.add_argument("--model-name", type=str, required=True)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--max-new-tokens", type=int, default=32)
    parser.add_argument(
        "--message", type=str, default="Tell me a story with more than 1000 words."
    )
    args = parser.parse_args()

    main()
