from flask import Flask,jsonify,request,render_template
from transformers import GPT2LMHeadModel, GPT2Tokenizer,Trainer, TrainingArguments,TextDataset, DataCollatorForLanguageModeling
import torch
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model_name_or_path = "sberbank-ai/rugpt3large_based_on_gpt2"
tokenizer = GPT2Tokenizer.from_pretrained(model_name_or_path)


model_name_or_path = "./models/itoglast"
model = GPT2LMHeadModel.from_pretrained(model_name_or_path).to(DEVICE)
app = Flask(__name__)

@app.route('/chg/<str>')
def user_profile(str):
    model = GPT2LMHeadModel.from_pretrained(f"./models/{str}").to(DEVICE)
    return f"Установлена модель: {model}"


def getans(text,beam=2,temp=1.9,min=50,max=100):
    input_ids = tokenizer.encode(text, return_tensors="pt").to(DEVICE)
    model.eval()
    with torch.no_grad():
        out = model.generate(input_ids, 
                            do_sample=True,
                            num_beams=int(beam),
                            temperature=float(temp),
                            #top_p=0.9,
                        #  top_k=50,
                            min_length= int(min),
                            max_length= int(max),
                            )

    generated_text = list(map(tokenizer.decode, out))[0]
    return generated_text

@app.route('/')
def ReturnJSON():
  return render_template('index.html')

@app.route('/send', methods = ['GET'])
def hello():
    data={
        "req": request.args.get('req'), 
        "ans": getans(request.args.get('req'),request.args.get('beam'),request.args.get('temp'),request.args.get('min'),request.args.get('max'))
    }
    reqs = request.args
    return jsonify(data)


if __name__ == '__main__':
    app.run(host="0.0.0.0",port=3001,debug=True)
