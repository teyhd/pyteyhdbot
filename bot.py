from transformers import GPT2LMHeadModel, GPT2Tokenizer,Trainer, TrainingArguments,TextDataset, DataCollatorForLanguageModeling
import torch
import telebot
import os
import re
import time

bot = telebot.TeleBot('token');
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model_name_or_path = "sberbank-ai/rugpt3large_based_on_gpt2"
tokenizer = GPT2Tokenizer.from_pretrained(model_name_or_path)
tokenizer.add_special_tokens({'pad_token': '[PAD]'})
hmodels = os.listdir('./models')
model_name_or_path = "./models/sma"
#model_name_or_path = "./models/me88"
model = GPT2LMHeadModel.from_pretrained(model_name_or_path).to(DEVICE)
#model.greedy_search
temp=0.25
def getans(text,beam=2,min=50,max=100,context=None):
    global temp
    input_ids = tokenizer.encode(text, return_tensors="pt",padding=True, max_length=512).to(DEVICE)
    model.eval()
    with torch.no_grad():
        out = model.generate(input_ids, 
                            do_sample=True,
                            num_beams=int(beam),
                          #  num_beam_groups =int(beam),
                            temperature=float(temp),
                            top_p=0.35,
                            top_k=95,
                            min_length= int(min),
                            max_length= int(max),
                            )

    generated_text = list(map(tokenizer.decode, out))[0]
    return generated_text

beam=2
#temp=1.9
min=1
max=2
delstr = True

beam=1

min=1
max=10
delstr = True
self = False

def msg(message):
        global beam,hmodels,temp,min,max,model,model_name_or_path,delstr,self
        who = message.from_user.first_name
        msguser = message.text
        context = None
        strnum =1
        if (message.reply_to_message!=None):
            strnum =2
            message.text = f"{message.reply_to_message.text}\n{message.text}"
        max=len(message.text)+3
        ans = getans(message.text,beam,min,max,context)
        
        olda = ans
        print(ans)
        ans = strnorm(ans,strnum)
        
        if delstr:
            ans = "".join(ans.split(message.text))

        smil =re.search("хах",ans)
        sm =re.search("Ахах",ans)
        ens =re.search("е знаю",ans) 
        print(smil,sm,ens)
        if smil or sm or ens:
            print(ans)
            temp =float(temp) + 0.05
            print(temp)
            ans = strnorm(ans,strnum+2)
            bot.send_message(message.from_user.id, ans)
            getans(message.text,beam,min,max,context)

        if ans=="":
            ans = olda
        bot.send_message(message.from_user.id, ans)
        if message.from_user.id!=304622290:
          bot.send_message(304622290, f"{who}\n{msguser}\n{ans}")
        if self:
            time.sleep(1)
            message.text = ans
            msg(message)

@bot.message_handler(content_types=['text'])
def start(message):
    global beam,hmodels,temp,min,max,model,model_name_or_path,delstr,self
    #print(message.reply_to_message.text)
    if message.text == '/beam':
        bot.send_message(message.from_user.id, 'Сколько лучей поставить?')
        bot.register_next_step_handler(message,set_beam)
    elif message.text == '/start':
        bot.send_message(message.from_user.id, '/mod - Сменить модель\n/beam - Сменить количество лучей\n/temp - Сменить температуру\n/min - Сменить минимальное количество символов\n/max - Сменить максимальное количество символов\n/del - Дублировать ваш ответ\nИли введите текст')
    elif message.text == '/temp':
        bot.send_message(message.from_user.id, 'Укажи новую температуру')
        bot.register_next_step_handler(message,set_temp)
    elif message.text == '/min':
        bot.send_message(message.from_user.id, 'Укажи минимальное количество символов')
        bot.register_next_step_handler(message,set_min)
    elif message.text == '/max':
        bot.send_message(message.from_user.id, 'Укажи максимальное количество символов')
        bot.register_next_step_handler(message,set_max)
    elif message.text == '/mod':
        bot.send_message(message.from_user.id, 'Введите номер модели?')
        hmodels = os.listdir('./models')
        string=''
        for x in range(0,len(hmodels)):
            string += f"{x}: {hmodels[x]}\n"
            bot.send_message(message.from_user.id, f"{x}: {hmodels[x]}\n")
        bot.register_next_step_handler(message,set_model)
    elif message.text == '/del':
        if delstr:
            delstr=False
        else:
            delstr=True
        bot.send_message(message.from_user.id, f"Удаление вашего сообщения: {delstr}")
    elif message.text == '/par':
        bot.send_message(message.from_user.id, f"Параметры нейронной сети:\n/mod - модель:{model_name_or_path}\n/beam - {beam}\n/temp - {temp}\n/min - {min}\n/max - {max}\n/del - {delstr}\n/self - {self}")
    elif message.text == '/self':
        if self:
            self=False
        else:
            self=True
        bot.send_message(message.from_user.id, f"Самоответ: {self}") 
    else:
        msg(message)
        
def strnorm(ans,strnum=1):
    olda = ans
    try:
      ans = "".join(ans.split("\n")[strnum])
    except:
        ans = ans
    if ans=="":
        ans = olda
   # ans = "".join(ans.split(".")[0])
    return ans

def set_model(message):
    global model,hmodels,model_name_or_path
    if isinstance(int(message.text), int):
        if int(message.text)>=0 and int(message.text)<=len(hmodels):
            model_name_or_path = hmodels[int(message.text)]
            model = GPT2LMHeadModel.from_pretrained(f"./models/{model_name_or_path}").to(DEVICE)
            bot.send_message(message.from_user.id,'Устарновленна модель: '+message.text)
            return True
    bot.send_message(message.from_user.id,'Нет модели с номером: '+message.text)
    bot.register_next_step_handler(message, start)

def set_beam(message):
    global beam
    beam = message.text
    bot.send_message(message.from_user.id,'Установил лучи: '+message.text)
    bot.register_next_step_handler(message, start)

def set_temp(message):
    global temp
    temp = message.text
    bot.send_message(message.from_user.id,'Установил температуру: '+temp)
    bot.register_next_step_handler(message, start)

def set_min(message):
    global min
    global max
    if int(message.text)<=max:
       min = int(message.text)
       bot.send_message(message.from_user.id,f'Установил минимально знаков: {min}')
    else:
        bot.send_message(message.from_user.id,f"Ошибка минимальное значение:{message.text} больше максимального:{max}")
    bot.register_next_step_handler(message, start)

def set_max(message):
    global max
    global min
    if int(message.text)>=min:
        max = int(message.text)
        bot.send_message(message.from_user.id,f'Установил максимально знаков: {max}')
    else:
        bot.send_message(message.from_user.id,f"Ошибка максимальное значение:{message.text} меньше минимального:{min}")
    bot.register_next_step_handler(message, start)

from urllib.request import urlopen

@bot.message_handler(content_types=['photo'])
def handle_photo(message):
    print(message.text)
    photo = message.photo[-1]
    file_info = bot.get_file(photo.file_id)
    downloaded_file = bot.download_file(file_info.file_path)
    save_path = 'photo.jpg'
    with open(save_path, 'wb') as new_file:
        new_file.write(downloaded_file)
    with open(save_path, 'rb') as photo:
      bot.send_photo(304622290, photo)
    bot.reply_to(message, 'Я не умею обрабатывать картинки')


@bot.message_handler(content_types=['document', 'video', 'audio', 'voice', 'sticker'])
def handle_file(message):
    print(message.text)
    file_info = bot.get_file(message.document.file_id)
    downloaded_file = bot.download_file(file_info.file_path)
    save_path = 'file' + message.document.file_name  # сохраняем файл с его исходным именем
    with open(save_path, 'wb') as new_file:
        new_file.write(downloaded_file)
    with open(save_path, 'rb') as photo:
      bot.send_photo(304622290, photo)
    bot.reply_to(message, 'Я не умею обрабатывать картинки')



bot.send_message(304622290, "Нейронная сеть запущена")
bot.polling(none_stop=True, interval=0)
bot.send_message(304622290, "Нейронная сеть убита")
