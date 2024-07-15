from generate_impression import ImpressionGenerator
prompt_path = "prompt_CN.txt"
findings_path = "./test.txt"
api_provider = "qianfan"
model = "ERNIE-4.0-8K-Latest" 
stream = True
max_tokens = 1024*2
temperature = 1e-10
top_p = None
outfile = "Your output file path.csv"
ig = ImpressionGenerator(api_provider)
ig.main(prompt_path,findings_path, model, stream, max_tokens, temperature, top_p, outfile)
