import argparse
import sys
import os
from dotenv import load_dotenv
from analyzer import Analyzer

def main():
    load_dotenv()

    parser = argparse.ArgumentParser(description="Production Code Checker Agent (HF Edition)")
    parser.add_argument("file", help="Path to the file to analyze")
    parser.add_argument("--model", default="Qwen/Qwen2.5-Coder-32B-Instruct", help="HF Model ID to use (default: Qwen/Qwen2.5-Coder-32B-Instruct)")
    parser.add_argument("--api-key", help="Hugging Face Token (optional, defaults to HF_TOKEN env var)")
    
    args = parser.parse_args()

    file_path = args.file
    if not os.path.exists(file_path):
        print(f"Error: File '{file_path}' not found.")
        sys.exit(1)

    try:
        with open(file_path, "r") as f:
            code_content = f.read()
    except Exception as e:
        print(f"Error reading fileHeader: {str(e)}")
        sys.exit(1)

    print(f"Analyzing {file_path} using {args.model}...")
    
    analyzer = Analyzer(api_key=args.api_key, model=args.model)
    analysis = analyzer.analyze_code(code_content)

    print("\n" + "="*50 + "\n")
    print(analysis)
    print("\n" + "="*50 + "\n")

    # Optional: Save to file if needed, but printing to stdout is standard for CLIs
    # with open(f"{file_path}.analysis.md", "w") as f:
    #     f.write(analysis)
    # print(f"Analysis saved to {file_path}.analysis.md")

if __name__ == "__main__":
    main()
