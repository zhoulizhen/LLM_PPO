from common import Tokenizer,ModelCLS,ModelGEN,ModelPPO,generate
import torch

device = 'cuda' if torch.cuda.is_available() else 'cpu'


tokenizer = Tokenizer()
[tokenizer.decode(i) for i in tokenizer.get_batch_data(prefix=True)[1]][:10]

model_ppo = torch.load('ppo.model')
model_ppo.to(device)
model_ppo.eval()

#随机一批数据
_, input_ids, _ = tokenizer.get_batch_data(prefix=True)

#切分成question和answer
split = [i.index(tokenizer.encoder['=']) + 1 for i in input_ids]
question = [input_ids[i][:split[i]] for i in range(len(input_ids))]
answer = [input_ids[i][split[i]:] for i in range(len(input_ids))]

#根据question生成predict
input_ids = [torch.LongTensor(i).unsqueeze(0).to(device) for i in question]
predict = [generate(model_ppo.model_gen, i) for i in input_ids]

#裁剪,只要生成的部分
predict = [p[0].tolist()[len(q):] for p, q in zip(predict, question)]

#解码成文本
question = [tokenizer.decode(i) for i in question]
answer = [tokenizer.decode(i) for i in answer]
predict = [tokenizer.decode(i) for i in predict]

acc = 0
for q, a, p in zip(question, answer, predict):
    print(q, a, p, a == p)
    if a == p:
        acc += 1

acc / len(question)
