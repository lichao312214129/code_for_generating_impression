from generate_impression import ImpressionGenerator
prompt_path = "prompt_CN.txt"
findings_path = "./test.txt"
api_provider = "uiuiapi" 
model = "claude35sonnet20240702"
outfile = "Your output file path.csv"
stream = True
max_tokens = 1024*2
temperature = 1e-10
top_p = None
ig = ImpressionGenerator(api_provider)
ig.main(prompt_path,findings_path, model, stream, max_tokens, temperature, top_p, outfile)
