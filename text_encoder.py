import torch
import clip
import os
import time

# Load CLIP model and preprocessing tools
device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load("ViT-B/32", device=device)


# Define function to read file and extract descriptions
def extract_descriptions(file_path):
    descriptions = []

    if not os.path.exists(file_path):
        print(f"File path does not exist: {file_path}")
        return descriptions

    with open(file_path, 'r', encoding='utf-8') as file:
        current_description = ""
        for line in file:
            line = line.strip()

            # Extract the part after "Model response:"
            if "Model response:" in line:
                response_part = line.split("Model response:")[1].strip()
                current_description += " " + response_part

            # When "Batch" keyword is encountered, save the previous description and start a new one
            if line.startswith("Batch"):
                if current_description:
                    descriptions.append(current_description.strip())
                current_description = ""

        # Process the last description
        if current_description:
            descriptions.append(current_description.strip())

    return descriptions


# Define wrapper function to extract text features and return results
def get_text_features(descriptions, batch_size=32):
    if not descriptions:
        print("No descriptions extracted!")
        return None

    text_features = []
    start_time = time.time()  # Record start time for extraction

    for i in range(0, len(descriptions), batch_size):
        batch_texts = descriptions[i:i + batch_size]
        text_input = clip.tokenize(batch_texts).to(device)
        with torch.no_grad():
            features = model.encode_text(text_input)

        # Adjust feature dimension to 768 if needed (original comment was 768, but code targets 512)
        # Assuming the target dimension based on the `size=(512,)` in the interpolate function.
        if features.size(1) != 512:  # If features are not 512-dimensional, convert
            features = torch.nn.functional.interpolate(features.unsqueeze(1), size=(512,), mode='linear', align_corners=False).squeeze(1)

        text_features.append(features)

    end_time = time.time()  # Record end time for extraction
    elapsed_time = end_time - start_time
    print(f"Total feature extraction time: {elapsed_time:.2f} seconds")

    if text_features:
        return torch.cat(text_features)
    else:
        print("No text features extracted!")
        return None