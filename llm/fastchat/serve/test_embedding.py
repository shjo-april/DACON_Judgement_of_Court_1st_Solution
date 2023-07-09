"""Send a test message."""
import argparse
import json

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

    conv = get_conversation_template(model_name)
    conv.append_message(conv.roles[0], args.message)
    conv.append_message(conv.roles[1], None)
    prompt = conv.get_prompt()

    headers = {"User-Agent": "FastChat Client"}
    gen_params = {
        "model": model_name,
        "input": prompt,
    }
    response = requests.post(
        worker_addr + "/worker_get_embeddings",
        headers=headers,
        json=gen_params,
        stream=False,
    )

    data = response.json()

    print(data.keys())
    input()

    print(f"{conv.roles[0]}: {args.message}")
    print(f"{conv.roles[1]}: ", end="")
    prev = 0
    for chunk in response.iter_lines(decode_unicode=False, delimiter=b"\0"):
        if chunk:
            data = json.loads(chunk.decode())
            output = data["text"].strip()
            print(output[prev:], end="", flush=True)
            prev = len(output)
    print("")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--controller-address", type=str, default="http://localhost:21001"
    )
    parser.add_argument("--worker-address", type=str)
    parser.add_argument("--model-name", type=str, default='vicuna-7b-v1.3')
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--max-new-tokens", type=int, default=32)
    parser.add_argument(
        "--message", type=str, default="Tell me a story with more than 1000 words."
    )
    args = parser.parse_args()

    # args.message = """
    # Decide whether the first party wins based on the facts of an event.
    # - The first party: Washington State Apple Advertising Commission
    # - The second party: Hunt
    # - The facts of an event: In 1972, the North Carolina Board of Agriculture adopted a regulation that required all apples shipped into the state in closed containers to display the USDA grade or nothing at all. Washington State growers (whose standards are higher than the USDA) challenged the regulation as an unreasonable burden to interstate commerce. North Carolina stated it was a valid exercise of its police powers to create "uniformity" to protect its citizenry from "fraud and deception."
    # """

    args.message = """
    Decide whether the first party wins based on the facts of an event.
    - The first party: Salerno
    - The second party: United States
    - The facts of an event: The 1984 Bail Reform Act allowed the federal courts to detain an arrestee prior to trial if the government could prove that the individual was potentially dangerous to other people in the community. Prosecutors alleged that Salerno and another person in this case were prominent figures in the La Cosa Nostra crime family.
    """

    main()
