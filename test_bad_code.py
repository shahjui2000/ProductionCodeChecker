import boto3
import torch

# Load model
model = torch.load("model.pth")
tokenizer = None # pretend we loaded this

def predict(text, max_len=100):
    # No error handling
    tokens = tokenizer.encode(text)
    
    # Hardcoded GPU device
    input_tensor = torch.tensor(tokens).cuda() 
    
    # Potential huge memory usage if max_len is large
    output = model.generate(input_tensor, max_length=max_len)
    
    return tokenizer.decode(output)

if __name__ == "__main__":
    # Hardcoded credentials
    s3 = boto3.client('s3', aws_access_key_id='AKIA...', aws_secret_access_key='secret...')
    
    print(predict("Hello world"))
