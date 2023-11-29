from common import Tokenizer,ModelCLS,ModelGEN,ModelPPO,generate
import torch

device = 'cuda' if torch.cuda.is_available() else 'cpu'


tokenizer = Tokenizer()
[tokenizer.decode(i) for i in tokenizer.get_batch_data(prefix=False)[1]][:10]

model_cls = ModelCLS()

optimizer = torch.optim.AdamW(params=model_cls.parameters(), lr=1e-4)
criterion = torch.nn.CrossEntropyLoss()

for epoch in range(500):
    print('epoch{}'.format(epoch))
    label, input_ids, attention_mask = tokenizer.get_batch_data(prefix=False)
    label = torch.LongTensor(label).to(device)
    input_ids = torch.LongTensor(input_ids).to(device)
    attention_mask = torch.LongTensor(attention_mask).to(device)

    logits = model_cls(input_ids=input_ids, attention_mask=attention_mask)

    loss = criterion(logits, label)
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()

    if epoch % 100 == 0:
        logits = logits.argmax(1)
        acc = (logits == label).sum().item() / len(label)
        print(epoch, acc)

        for i in range(2):
            print(tokenizer.decode(input_ids[i].tolist()), logits[i].item())

model_cls.to('cpu')
torch.save(model_cls, 'cls.model')